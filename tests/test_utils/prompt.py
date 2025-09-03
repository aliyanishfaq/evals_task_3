LLM_AS_A_JUDGE_PROMPT = '''
You are an expert coding evaluator for coding agent that researches company information. You will be provided with a coding agent's implementation and an expert-written implementation that represents the gold standard.

Your job is to evaluate the coding agent's implementation for correctness, code quality, and adherence to the expert's architectural and code cleanliness philosophy. 
You will be provided with basic requirements and good practices. You need to check for presence of each of the basic requirements and good practices.
Additionally, you will be asked to evaluate the code for correctness and quality. For correctness error, you will be expected to add an entry to the code_correctness_evidence array.

**CRITICAL RULE - NO DUPLICATE ISSUES**: Never duplicate the same issue across multiple evidence arrays. Each specific issue should only appear ONCE in the entire evaluation - either in code_correctness_evidence OR code_quality_evidence, but never both. If an issue affects both correctness and quality, choose the most appropriate category and only include it there.

**CRITICAL RULE - NO CONTRADICTIONS**: If you mark a basic requirement as `true`, do NOT add contradictory evidence claiming it's missing or broken. The basic requirements and evidence arrays must be logically consistent.

Return your evaluation as a single JSON object.

## Evaluation Criteria

Basic Requirements: The code must meet these core specifications for a functional Company Researcher agent. Mark `true` if present and working correctly, `false` if absent or broken.

-  max_search_queries: The code should have a way to set the maximum number of search queries to be generated per requested.
-  max_search_results: The code should have a way to set the maximum number of search results to be generated per requested.
-  max_reflection_steps: The code should have a way to set the maximum number of reflection steps allowed to avoid infinite cycles.
-  reflection_step: The code should have a way to reflect on the completeness of the research and determine if additional searches are needed.



Good Practices: Additional qualities of a well-designed Company Researcher agent. Mark `true` if implemented correctly, `false` if absent.

- functional_reflection: The reflection node is connected to the web search node so that if the research is not complete, the agent can execute web searches again. The implementation should also be correct in terms of logic and the paths it takes.
- no_redundant_files: There should be no redundant files in the code. The code should be as clean and concise as possible. Return false if there are files like `test_agent.py`, `demo.py`
- preferred_state_type: The code should use TypedDict for the state. Return True if the code uses TypedDict for the state. If it uses something else, return false.
- separate_prompts: Expert code has separate prompts for query writing, reflection, and extraction. The code should have separate prompts for each of these. Return true if the code has separate prompts for each of these. Otherwise, return false.


## Evidence Categorization Guidelines

**Code Quality Evidence** (architectural/design issues):
- No separation of features. Each feature should be in a separate node.
- Unnecessary complexity
- Correct primitives for Langgraph types such as state, messages (which should be defined as Annotated[list, add_messages] in state)

**Code Correctness Evidence** (functional/runtime bugs):
- Runtime exceptions
- Type errors, import failures

**Remember**: NEVER put the same issue in both categories. Choose the most appropriate one.

## JSON Response Structure

{{
  "basic_requirements": {{
    "max_search_queries": boolean,
    "max_search_results": boolean,
    "max_reflection_steps": boolean,
    "reflection_step": boolean,
  }},
  "good_practices": {{
    "functional_reflection": boolean,
    "no_redundant_files": boolean,
    "preferred_state_type": boolean,
    "separate_prompts": boolean,
  }},
  "code_quality_check": boolean,
  "code_quality_evidence": [
    {{ "issue": string, "severity": "minor|major|critical" }}
  ],
  "code_correctness_check": boolean,
  "code_correctness_evidence": [
    {{ "issue": string, "severity": "minor|major|critical" }}
  ]
}}

Task given to coding agent:
{user_task}

Expert implementation (gold standard):
{expert_code}

Coding agent implementation to evaluate:
{user_code}

The human annotator added the following notes when running the code:
{human_notes}
'''

USER_TASK = '''
Create me a company researcher built using langgraph. It should be a multi-node graph. The user should be expected to provide the name of the company and the optional notes if they want. There should be a set maximum search queries that we should do per company and max search results. The LLM should generate the queries that should be searched using the Tavily API to fill the following structured object

"title": "CompanyInfo", "description": "Basic information about a company", "type": "object", "properties": { "company_name": {"type": "string", "description": "Official name of the company"}, "founding_year": {"type": "integer", "description": "Year the company was founded"}, "founder_names": {"type": "array", "items": {"type": "string"}, "description": "Names of the founding team members"}, "product_description": {"type": "string", "description": "Brief description of the company's main product or service"}, "funding_summary": {"type": "string", "description": "Summary of the company's funding history”} "notable_customers": {"type": "string", "description": "Known customers that use company's product/service"}

}, "required": ["company_name"] }

There should be a reflection step at the end. This step should determine if we have good and sufficient information for the the company. If we don’t have sufficient information we should execute web searches again to get information on the company. There should also be a cap on the number of allowed reflection steps. Keep track of the conversation in messages array and try to do web searchers in parallel to improve speed.
'''

EXPERT_CODE = '''
File Name: configuration.py

-----------------------------

File Content: 

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    max_search_queries: int = 3  # Max search queries per company
    max_search_results: int = 3  # Max search results per query
    max_reflection_steps: int = 0  # Max reflection steps
    include_search_results: bool = (
        False  # Whether to include search results in the output
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {{}}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{{k: v for k, v in values.items() if v}})




File Name: graph.py

-----------------------------

File Content: 

import asyncio
from typing import cast, Any, Literal
import json

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_sources, format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Search

tavily_async_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


async def research_company(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Search tasks
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    deduplicated_search_docs = deduplicate_sources(search_docs)
    source_str = format_sources(
        deduplicated_search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(p)
    state_update = {
        "completed_notes": [str(result.content)],
    }
    if configurable.include_search_results:
        state_update["search_results"] = deduplicated_search_docs

    return state_update


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()




File Name: prompts.py

-----------------------------

File Content: 

EXTRACTION_PROMPT = """Your task is to take notes gathered from web research and extract them into the following schema.

<schema>
{{info}}
</schema>

Here are all the notes from research:

<web_research_notes>
{{notes}}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific company information.

Here is the company you are researching: {{company}}

Generate at most {{max_search_queries}} search queries that will help gather the following information:

<schema>
{{info}}
</schema>

<user_notes>
{{user_notes}}
</user_notes>

Your query should:
1. Focus on finding factual, up-to-date company information
2. Target official sources, news, and reliable business databases
3. Prioritize finding information that matches the schema requirements
4. Include the company name and relevant business terms
5. Be specific enough to avoid irrelevant results

Create a focused query that will maximize the chances of finding schema-relevant information."""

INFO_PROMPT = """You are doing web research on a company, {{company}}. 

The following schema shows the type of information we're interested in:

<schema>
{{info}}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about the company, focusing on topics relevant to our interests.

<Website contents>
{{content}}
</Website contents>

Here are any additional notes from the user:
<user_notes>
{{user_notes}}
</user_notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

REFLECTION_PROMPT = """You are a research analyst tasked with reviewing the quality and completeness of extracted company information.

Compare the extracted information with the required schema:

<Schema>
{{schema}}
</Schema>

Here is the extracted information:
<extracted_info>
{{info}}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:
1. Are any required fields missing?
2. Are any fields incomplete or containing uncertain information?
3. Are there fields with placeholder values or "unknown" markers?
"""




File Name: state.py

-----------------------------

File Content: 

from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator


DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CompanyInfo",
    "description": "Basic information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official name of the company",
        },
        "founding_year": {
            "type": "integer",
            "description": "Year the company was founded",
        },
        "founder_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the founding team members",
        },
        "product_description": {
            "type": "string",
            "description": "Brief description of the company's main product or service",
        },
        "funding_summary": {
            "type": "string",
            "description": "Summary of the company's funding history",
        },
    },
    "required": ["company_name"],
}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    search_results: list[dict] = field(default=None)
    "List of search results"

    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    search_results: list[dict] = field(default=None)
    "List of search results"




File Name: utils.py

-----------------------------

File Content: 

def deduplicate_sources(search_response: dict | list[dict]) -> list[dict]:
    """
    Takes either a single search response or list of responses from Tavily API and de-duplicates them based on the URL.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_urls = set()
    unique_sources_list = []
    for source in sources_list:
        if source["url"] not in unique_urls:
            unique_urls.add(source["url"])
            unique_sources_list.append(source)

    return unique_sources_list


def format_sources(
    sources_list: list[dict],
    include_raw_content: bool = True,
    max_tokens_per_source: int = 1000,
) -> str:
    """
    Takes a list of unique results from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        sources_list: list of unique results from Tavily API
        max_tokens_per_source: int, maximum number of tokens per each search result to include in the formatted string
        include_raw_content: bool, whether to include the raw_content from Tavily in the formatted string

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Format output
    formatted_text = "Sources:\n\n"
    for source in sources_list:
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {{max_tokens_per_source}} tokens: {{raw_content}}\n\n"

    return formatted_text.strip()


def format_all_notes(completed_notes: list[str]) -> str:
    """Format a list of notes into a string"""
    formatted_str = ""
    for idx, company_notes in enumerate(completed_notes, 1):
        formatted_str += f"""
{'='*60}
Note: {idx}:
{'='*60}
Notes from research:
{company_notes}"""
    return formatted_str



'''
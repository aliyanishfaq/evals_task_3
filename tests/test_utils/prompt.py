LLM_AS_A_JUDGE_PROMPT = '''
You are an expert coding evaluator for text-to-SQL agent implementations. You will be provided with a coding agent's implementation and an expert-written implementation that represents the gold standard.

Your job is to evaluate the coding agent's implementation for correctness, code quality, and adherence to the expert's architectural philosophy. 
You will be provided with basic requirements and good practices. You need to check for presence of each of the basic requirements and good practices.
Additionally, you will be asked to evaluate the code for correctness and quality. For correctness error, you will be expected to add an entry to the code_correctness_evidence array.
For code quality issues, you will be expected to add an entry to the code_quality_evidence array. Remember to not add any entry more than once. Determine the severity of the issue and if an
issue has been used to flag something earlier in basic requirements or good practices, you should not add it again and ignore that.

Return your evaluation as a single JSON object.

## Evaluation Criteria

Basic Requirements: The code must meet these core specifications for a functional SQL agent. Mark `true` if present and working correctly, `false` if absent or broken.

- database_initialization: Correctly fetches Chinook database from GitHub and sets up SQLite connection
- schema_extraction: Correctly extracts database schema information for SQL generation context  
- sql_generation: Correctly converts natural language queries to valid SQL using LLM
- query_execution: Correctly executes SQL against database and captures results
- natural_language_response: Correctly returns human-readable answers based on query results as AIMessage
- irrelevant_query_handling: Correctly responds "I don't know" or similar for non-database questions

Good Practices: Additional qualities of a well-designed SQL agent. Mark `true` if implemented correctly, `false` if absent.

- **separation_of_concerns**: Distinct nodes/functions for SQL generation, execution, and response formatting
- **error_handling**: Gracefully handles SQL errors, connection issues, and malformed queries

Expert code uses a clean linear pipeline: generate_sql → execute_sql → generate_answer. Each node has a single responsibility with direct database integration (no tools). Evaluate alignment with this philosophy.
When evaluating code quality, you should check for the patterns used in the expert code. If there is any unecessary complexity or if there is a significant difference from the expert's code architecture, you should
flag that.

## JSON Response Structure

{{
  "basic_requirements": {{
    "database_initialization": boolean,
    "schema_extraction": boolean, 
    "sql_generation": boolean,
    "query_execution": boolean,
    "natural_language_response": boolean,
    "irrelevant_query_handling": boolean
  }},
  "good_practices": {{
    "separation_of_concerns": boolean,
    "error_handling": boolean,
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
Create a LangGraph-based text-to-SQL agent. It should generate sql, execute that sql against the chinook sqlite database, and then generate a response answer in natural language based on the result.
To use chinook db, fetch it from `https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql` and create  in-memory SQLite database using langgraph
It might be useful to pass in the detailed info of the table schema to the prompt so the agent can convert user's natural request to correct sql. Whenever, the query is irrelevant, or cannot be answered using
the sql db search, just say you don't know the answer and don't talk about anything. Your purpose is to only convert text request to sql and generate response in natural language.
'''

EXPERT_CODE = '''
prompts.py

QA_SYSTEM_PROMPT = """
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: How many songs do you have by James Brown?
Context:[20]
Helpful Answer: We have 20 songs by James Brown.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
You will have the full message history to help you answer the question, if you need more information, ask the user for it.
"""

SQL_SYSTEM_PROMPT = """
Task: Generate SQL statement to query a database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a SQL statement.
Do not include any text except the generated SQL statement.
You will have the full message history to help you answer the question, if you dont need to generate a sql query, just generate a sql query that will return an empty result.
"""

utils.py

import sqlite3

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool


# fetch the chinook database from github and create an in-memory database
def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url, timeout=10)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def get_db_table_names():
    """Get list of all table names in the database."""
    engine = get_engine_for_chinook_db()
    db = SQLDatabase(engine)
    return db.get_usable_table_names()


def get_detailed_table_info():
    """Get detailed information for each table including schema, keys, and sample data."""
    engine = get_engine_for_chinook_db()
    db = SQLDatabase(engine)
    inspector = inspect(engine)
    table_names = db.get_usable_table_names()

    detailed_info = {{}}

    for table_name in table_names:
        table_info = {
            "columns": [],
            "primary_key": None,
            "foreign_keys": [],
            "sample_data": [],
        }

        # Get table schema using SQLAlchemy inspector
        try:
            columns = inspector.get_columns(table_name)
            for column in columns:
                table_info["columns"].append(
                    {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get("nullable", "unknown"),
                    }
                )

            # Get primary key
            pk = inspector.get_pk_constraint(table_name)
            if pk["constrained_columns"]:
                table_info["primary_key"] = pk["constrained_columns"]

            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                table_info["foreign_keys"].append(
                    {
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                    }
                )

        except Exception as e:
            table_info["error"] = str(e)

        # Get sample data (first 3 rows)
        try:
            sample_query = f"SELECT * FROM {{table_name}} LIMIT 3"  # nosec B608
            sample_result = db.run(sample_query)
            table_info["sample_data"] = sample_result
        except Exception as e:
            table_info["sample_data_error"] = str(e)

        detailed_info[table_name] = table_info

    return detailed_info


def get_schema_overview():
    """Get a concise overview of all table schemas."""
    engine = get_engine_for_chinook_db()
    db = SQLDatabase(engine)
    inspector = inspect(engine)
    table_names = db.get_usable_table_names()

    schema_overview = {{}}

    for table_name in table_names:
        try:
            columns = inspector.get_columns(table_name)
            schema_overview[table_name] = [
                {"name": col["name"], "type": str(col["type"])} for col in columns
            ]
        except Exception as e:
            schema_overview[table_name] = {"error": str(e)}

    return schema_overview


# Example usage
if __name__ == "__main__":
    print("=== Basic Table Names ===")
    table_names = get_db_table_names()
    print(table_names)
    print()

    print("=== Detailed Table Information ===")
    detailed_info = get_detailed_table_info()
    for table_name, info in detailed_info.items():
        print(f"--- Table: {{table_name}} ---")

        if "error" in info:
            print(f"Error: {info['error']}")
        else:
            print("Columns:")
            for col in info["columns"]:
                print(f"  - {col['name']}: {col['type']} (nullable: {col['nullable']})")

            if info["primary_key"]:
                print(f"  Primary Key: {info['primary_key']}")

            if info["foreign_keys"]:
                print("  Foreign Keys:")
                for fk in info["foreign_keys"]:
                    print(
                        f"    {fk['columns']} -> {fk['referred_table']}.{fk['referred_columns']}"
                    )

        if "sample_data_error" in info:
            print(f"Sample data error: {info['sample_data_error']}")
        else:
            print(f"Sample data: {info['sample_data']}")

        print("-" * 50)

    print("=== Database Schema Overview ===")
    schema_overview = get_schema_overview()
    for table_name, columns in schema_overview.items():
        print(f"{table_name}:")
        if isinstance(columns, list):
            for column in columns:
                print(f"  {column['name']}: {column['type']}")
        else:
            print(f"  Error: {columns['error']}")


simple_text2sql.py


from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, List, TypedDict

import sys
import os
from pathlib import Path

# Add current directory to path for imports when loaded as standalone module
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from prompts import QA_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT
from utils import get_detailed_table_info, get_engine_for_chinook_db

load_dotenv(override=True)


class OverallState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    schema: str
    sql: str
    records: List[dict]


class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class OutputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def generate_sql(llm):
    def _generate(state: OverallState) -> dict:
        last_message = state["messages"][-1]
        prompt = f"""Generate a SQL query for the following question:
        Question: {{last_message.content}}
        Schema: {{get_detailed_table_info()}}
        SQL:
        """
        sql_query = llm.invoke(
            [SystemMessage(SQL_SYSTEM_PROMPT)]
            + state["messages"]
            + [HumanMessage(prompt)]
        )
        sql_query = sql_query.content.replace("```sql", "").replace("```", "")
        return {"sql": sql_query}

    return _generate


def execute_sql(db):
    def _execute(state: OverallState) -> dict:
        records = db.run(state["sql"])
        return {"records": records}

    return _execute


def generate_answer(llm):
    def _answer(state: OverallState) -> dict:
        last_message = state["messages"][-1]
        prompt = f"Given the question: {last_message.content} and the database results: {state['records']}, provide a concise answer."
        answer = llm.invoke(
            [SystemMessage(QA_SYSTEM_PROMPT)]
            + state["messages"]
            + [HumanMessage(prompt)]
        )
        return {"messages": [answer]}

    return _answer


def create_agent(llm, db):
    builder = StateGraph(
        OverallState, input_schema=InputState, output_schema=OutputState
    )
    builder.add_node("generate_sql", generate_sql(llm))
    builder.add_node("execute_sql", execute_sql(db))
    builder.add_node("generate_answer", generate_answer(llm))
    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_edge("execute_sql", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
db = SQLDatabase(get_engine_for_chinook_db())
agent = create_agent(llm, db)

# Export as 'app' as required by the test framework
app = agent
'''
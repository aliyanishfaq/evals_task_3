import os, json, pathlib, importlib.util, sys, hashlib, pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import cast, List, Literal
from test_utils.prompt import LLM_AS_A_JUDGE_PROMPT, USER_TASK, EXPERT_CODE
from test_utils.format_code import folder_to_prompt_string
from test_utils.git_branch import get_git_branch

CANDIDATE_NAME = get_git_branch()
LLM_AS_JUDGE_MODEL = "claude-sonnet-4-20250514"
CODE_FOLDER = [pathlib.Path("../")]

HUMAN_NOTES = """
"""

class LlmAsJudgeEvidence(BaseModel):
    issue: str
    severity: Literal["minor", "major", "critical"]

class BasicRequirements(BaseModel):
    database_initialization: bool # 1 points
    schema_extraction: bool # 1 points
    sql_generation: bool # 1 points
    query_execution: bool # 1 points
    natural_language_response: bool # 1 points
    irrelevant_query_handling: bool # 1 points

class GoodPractices(BaseModel):
    separation_of_concerns: bool # 1 points
    error_handling: bool # 1 points

class LlmAsJudgeOutput(BaseModel):
    basic_requirements: BasicRequirements
    good_practices: GoodPractices
    code_quality_check: bool
    code_quality_evidence: List[LlmAsJudgeEvidence]
    code_correctness_check: bool
    code_correctness_evidence: List[LlmAsJudgeEvidence]

def _write_score(score):
    out = pathlib.Path("results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"code_quality_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, awarded_pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": awarded_pts, "passed": ok, "msg": msg})
    score["points"] += awarded_pts

def _load_judge():
    """
    Returns a (invoke, model_name) tuple.
    """
    llm = ChatAnthropic(model=LLM_AS_JUDGE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(LlmAsJudgeOutput)

    return (lambda msgs: structured_llm.invoke(msgs)), f"anthropic:{LLM_AS_JUDGE_MODEL}"

def _calculate_score(evidence_list: List[LlmAsJudgeEvidence], max_points: int) -> float:
    """Calculates a score based on a list of evidence items and their severity."""
    points_deducted = 0
    for evidence in evidence_list:
        if evidence.severity == "critical":
            points_deducted += 2
        elif evidence.severity == "major":
            points_deducted += 1
        elif evidence.severity == "minor":
            points_deducted += 0.5
        else:
            print(f"Warning: Unknown severity level '{evidence.severity}'")
            points_deducted += 1
    
    return max(0, max_points - points_deducted)

def test_best_practices_llm_judge():
    score = {"candidate": CANDIDATE_NAME, "bucket": "code_quality", "points": 0, "max_points": 20, "details": []}
    user_code = folder_to_prompt_string(CODE_FOLDER)

    with open('txt_dump/user_code.txt', 'w') as f:
        f.write(user_code)

    # Prompt the judge with task-specific guidelines
    system = LLM_AS_A_JUDGE_PROMPT.format(user_task=USER_TASK, expert_code=EXPERT_CODE, user_code=user_code, human_notes=HUMAN_NOTES)
    user = {
        "role": "user",
        "content": "Return the JSON object evaluating the codebase."
    }

    try:
        invoke, model_name = _load_judge()
        resp = invoke([SystemMessage(content=system), HumanMessage(content=user["content"])])
        judge = cast(LlmAsJudgeOutput, resp)
    except Exception as e:
        _add(score, 0, "judge_error", False, f"Judge error: {type(e).__name__}: {e}")
        _write_score(score)
        pytest.fail(f"LLM judge failed: {e}")
        
    # Code Quality check
    code_quality_points = _calculate_score(judge.code_quality_evidence, 4)
    _add(score, code_quality_points, "code_quality_check", judge.code_quality_check, str(judge.code_quality_evidence))

    # Code Correctness check (for general bugs)
    code_correctness_points = _calculate_score(judge.code_correctness_evidence, 8)
    _add(score, code_correctness_points, "code_correctness_check", judge.code_correctness_check, str(judge.code_correctness_evidence))

    # Basic Requirements check
    _add(score, 1 if judge.basic_requirements.database_initialization else 0, "database_initialization", judge.basic_requirements.database_initialization, "Database initialization is present" if judge.basic_requirements.database_initialization else "Database initialization is not present")
    _add(score, 1 if judge.basic_requirements.schema_extraction else 0, "schema_extraction", judge.basic_requirements.schema_extraction, "Schema extraction is present" if judge.basic_requirements.schema_extraction else "Schema extraction is not present")
    _add(score, 1 if judge.basic_requirements.sql_generation else 0, "sql_generation", judge.basic_requirements.sql_generation, "SQL generation is present" if judge.basic_requirements.sql_generation else "SQL generation is not present")
    _add(score, 1 if judge.basic_requirements.query_execution else 0, "query_execution", judge.basic_requirements.query_execution, "Query execution is present" if judge.basic_requirements.query_execution else "Query execution is not present")
    _add(score, 1 if judge.basic_requirements.natural_language_response else 0, "natural_language_response", judge.basic_requirements.natural_language_response, "Natural language response is present" if judge.basic_requirements.natural_language_response else "Natural language response is not present")
    _add(score, 1 if judge.basic_requirements.irrelevant_query_handling else 0, "irrelevant_query_handling", judge.basic_requirements.irrelevant_query_handling, "Irrelevant query handling is present" if judge.basic_requirements.irrelevant_query_handling else "Irrelevant query handling is not present")

    # Good Practices check
    _add(score, 1 if judge.good_practices.separation_of_concerns else 0, "separation_of_concerns", judge.good_practices.separation_of_concerns, "Separation of concerns is present" if judge.good_practices.separation_of_concerns else "Separation of concerns is not present")
    _add(score, 1 if judge.good_practices.error_handling else 0, "error_handling", judge.good_practices.error_handling, "Error handling is present" if judge.good_practices.error_handling else "Error handling is not present")

    _write_score(score)
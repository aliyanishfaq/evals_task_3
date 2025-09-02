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
    max_search_queries: bool # 1 points
    max_search_results: bool # 1 points
    max_reflection_steps: bool # 1 points
    reflection_step: bool # 1 points

class GoodPractices(BaseModel):
    functional_reflection: bool # 2 points
    no_redundant_files: bool # 2 points
    preferred_state_type: bool # 1 points
    separate_prompts: bool # 1 points

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
    score = {"candidate": CANDIDATE_NAME, "bucket": "code_quality", "points": 0, "max_points": 22, "details": []}
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
    _add(score, 1 if judge.basic_requirements.max_search_queries else 0, "max_search_queries", judge.basic_requirements.max_search_queries, "Max search queries is present" if judge.basic_requirements.max_search_queries else "Max search queries is not present")
    _add(score, 1 if judge.basic_requirements.max_search_results else 0, "max_search_results", judge.basic_requirements.max_search_results, "Max search results is present" if judge.basic_requirements.max_search_results else "Max search results is not present")
    _add(score, 1 if judge.basic_requirements.max_reflection_steps else 0, "max_reflection_steps", judge.basic_requirements.max_reflection_steps, "Max reflection steps is present" if judge.basic_requirements.max_reflection_steps else "Max reflection steps is not present")
    _add(score, 1 if judge.basic_requirements.reflection_step else 0, "reflection_step", judge.basic_requirements.reflection_step, "Reflection step is present" if judge.basic_requirements.reflection_step else "Reflection step is not present")

    # Good Practices check
    _add(score, 2 if judge.good_practices.functional_reflection else 0, "functional_reflection", judge.good_practices.functional_reflection, "Functional reflection is present" if judge.good_practices.functional_reflection else "Functional reflection is not present")
    _add(score, 2 if judge.good_practices.no_redundant_files else 0, "no_redundant_files", judge.good_practices.no_redundant_files, "No redundant files is present" if judge.good_practices.no_redundant_files else "No redundant files is not present")
    _add(score, 1 if judge.good_practices.preferred_state_type else 0, "preferred_state_type", judge.good_practices.preferred_state_type, "Preferred state type is present" if judge.good_practices.preferred_state_type else "Preferred state type is not present")
    _add(score, 1 if judge.good_practices.separate_prompts else 0, "separate_prompts", judge.good_practices.separate_prompts, "Separate prompts is present" if judge.good_practices.separate_prompts else "Separate prompts is not present")

    _write_score(score)
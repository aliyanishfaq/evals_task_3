# Langgraph.json, llm requests, database integrity, simple query with processing, join query tests, date range query test, reject irrelevant queries
import os
import json
import pathlib
import importlib.util
import sys
import hashlib
import urllib.parse
import pytest
from langchain_core.messages import HumanMessage
from test_utils.test_state import INITIAL_STATE, TEST_STATES
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from test_utils.git_branch import get_git_branch

class LLMBinaryJudge(BaseModel):
    match: bool
    reasoning: str


CANDIDATE_NAME = get_git_branch()
DEFAULT_CONFIG_PATH = pathlib.Path.cwd() / "../langgraph.json"
DEFAULT_AGENT_PATH = pathlib.Path.cwd() / "../text_to_sql_agent.py"

def _write_score(score):
    out = pathlib.Path("results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"basic_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": (pts if ok else 0), "passed": bool(ok), "msg": msg})
    if ok: score["points"] += pts

def _load_module(agent_py_path: pathlib.Path):
    module_name = "candidate_agent_" + hashlib.md5(str(agent_py_path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(module_name, str(agent_py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_app(mod):
    if not hasattr(mod, "app"):
        raise AssertionError("agent.py must export a global variable `app`")
    return mod.app

def __llm_as_judge(test_response, expected_response):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    structured_llm = llm.with_structured_output(LLMBinaryJudge)
    prompt = f"""
    You are an evaluator. You will be provided with two responses: an Expected response (gold) and a Test response.
    Your task is to decide whether the Test response covers all of the information in the Expected response. 

    Key rules:
    - If the Test response contains all the information in the Expected response (regardless of extra details, formatting differences, or rephrasing), return True.
    - If the Test response is missing any information that appears in the Expected response, or if it contradicts the expected response, return False.
    - Focus on information completeness and correctness relative to the Expected response, not strict textual identity.

    Return a JSON object with the following fields:
    - match: bool  (True if the Test response fully covers the Expected response, False otherwise)
    - reasoning: str (Explain your reasoning in detail)

    Test response: {test_response}
    Expected response: {expected_response}
    """

    response = structured_llm.invoke(prompt)
    return response.match, response.reasoning

def _out_parser(out):
    return out["final_response"]

LLM_HOST_ALLOWLIST = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "openai.azure.com",
    "api.cohere.ai", "cohere.ai",
]

# unverified code
class HttpSpy:
    def __init__(self, monkeypatch):
        self.urls = []

        # httpx (most modern SDKs, incl. OpenAI v1 & Anthropic)
        try:
            import httpx
            # Sync
            if hasattr(httpx, "Client"):
                _orig_req = httpx.Client.request
                _orig_send = httpx.Client.send
                def _wrap_req(client, method, url, *a, **k):
                    self.urls.append(url); return _orig_req(client, method, url, *a, **k)
                def _wrap_send(client, request, *a, **k):
                    try:
                        self.urls.append(str(request.url))
                    except Exception:
                        pass
                    return _orig_send(client, request, *a, **k)
                monkeypatch.setattr(httpx.Client, "request", _wrap_req, raising=False)
                monkeypatch.setattr(httpx.Client, "send", _wrap_send, raising=False)
            # Async
            if hasattr(httpx, "AsyncClient"):
                _orig_areq = httpx.AsyncClient.request
                _orig_asend = httpx.AsyncClient.send
                async def _wrap_areq(client, method, url, *a, **k):
                    self.urls.append(url); return await _orig_areq(client, method, url, *a, **k)
                async def _wrap_asend(client, request, *a, **k):
                    try:
                        self.urls.append(str(request.url))
                    except Exception:
                        pass
                    return await _orig_asend(client, request, *a, **k)
                monkeypatch.setattr(httpx.AsyncClient, "request", _wrap_areq, raising=False)
                monkeypatch.setattr(httpx.AsyncClient, "send", _wrap_asend, raising=False)
        except Exception:
            pass

        try:
            import requests
            _orig_req = requests.sessions.Session.request
            def _wrap_req(session, method, url, *a, **k):
                self.urls.append(url); return _orig_req(session, method, url, *a, **k)
            monkeypatch.setattr("requests.sessions.Session.request", _wrap_req, raising=False)
        except Exception:
            pass


    def llm_calls(self):
        hits = []
        with open("results/urls.txt", "w") as f:
            f.write(str(self.urls))
        for url in self.urls:
            try:
                host = urllib.parse.urlparse(url).netloc
            except Exception:
                continue
            if any(h in host for h in LLM_HOST_ALLOWLIST):
                hits.append(url)
        return hits

def test_basics(monkeypatch):
    score = {"candidate": CANDIDATE_NAME, "bucket": "basic", "points": 0, "max_points": 22, "details": []}
    failures = []

    # A) Config validation (4 pts) — soft: record and continue
    try:
        import langgraph_cli.config as lgc
        cfg_env = os.getenv("LG_CONFIG_PATH", "").strip()
        cfg_path = pathlib.Path(cfg_env) if cfg_env else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        lgc.validate_config_file(cfg_path)  # raises on invalid config/spec
        _add(score, 4, "config_validate", True, "")
    except Exception as e:
        _add(score, 4, "config_validate", False, f"{type(e).__name__}: {e}")
        failures.append(f"Config validation failed: {e}")

    # B) Real LLM call happens during invoke (2 pts) — soft: record and continue
    try:
        agent_path_env = os.getenv("CANDIDATE_AGENT_PATH", "").strip()
        agent_py_path = pathlib.Path(agent_path_env) if agent_path_env else DEFAULT_AGENT_PATH
        if not agent_py_path.exists():
            raise FileNotFoundError(f"agent.py not found at: {agent_py_path}")
        mod = _load_module(agent_py_path)
        app = _get_app(mod)
        spy = HttpSpy(monkeypatch)
        initial_state = INITIAL_STATE
        out = app.invoke(initial_state)
        with open("txt_dump/validate_llm_call.txt", "w") as f:
            f.write(str(out))
        hits = spy.llm_calls()
        with open("txt_dump/llm_calls.txt", "w") as f:
            f.write(str(hits))
        ok = len(hits) > 0
        _add(score, 2, "llm_network_call", ok,
             "" if ok else "no outbound calls to known LLM providers observed")
        if not ok:
            failures.append("No outbound LLM HTTP calls observed during invoke.")
    except Exception as e:
        _add(score, 2, "llm_network_call", False, f"invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Invoke/LLM check failed: {e}")

    # C) Database integrity tests (6 pts total)
    db_integrity_score = 0
    db_integrity_failures = []
    
    # C.1) Artist count test (2 pts)
    try:
        state = TEST_STATES["c_1"]
        out = app.invoke(state)
        with open("txt_dump/artist_count_test.txt", "w") as f:
            f.write(str(out))
        response = _out_parser(out)

        ok = "275" in response
        if ok:
            db_integrity_score += 2
        else:
            db_integrity_failures.append("Artist count test: response does not contain 275")
    except Exception as e:
        db_integrity_failures.append(f"Artist count test failed: {e}")

    # C.2) Album existence test (2 pts)
    try:
        state = TEST_STATES["c_2"]
        out = app.invoke(state)
        with open("txt_dump/album_existence_test.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = "347" in response
        if ok:
            db_integrity_score += 2
        else:
            db_integrity_failures.append("Album existence test: response does not contain 347")
    except Exception as e:
        db_integrity_failures.append(f"Album existence test failed: {e}")

    # C.3) Highest track price test (2 pts)
    try:
        state = TEST_STATES["c_3"]
        out = app.invoke(state)
        with open("txt_dump/highest_price_test.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = "1.99" in response
        if ok:
            db_integrity_score += 2
        else:
            db_integrity_failures.append("Highest price test: response does not contain 1.99")
    except Exception as e:
        db_integrity_failures.append(f"Highest price test failed: {e}")

    # Add combined database integrity score
    _add(score, db_integrity_score, "database_integrity_tests", db_integrity_score != 0,
         f"Passed {db_integrity_score}/6 points" if db_integrity_score != 0 else "All database integrity tests failed")
    if db_integrity_failures:
        failures.extend(db_integrity_failures)

    # D.1) Simple query with processing
    try:
        state = TEST_STATES["d_1"]
        out = app.invoke(state)
        with open("txt_dump/simple_query_with_processing.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok, reasoning = __llm_as_judge(response, "The average invoice total per country, excluding countries with less than 5 invoices.")
        _add(score, 2, "simple_query_with_processing", ok,
             f"response: {response}\nreasoning: {reasoning}")
        if not ok:
            failures.append(f"Response does not contain the expected information: {response}\nreasoning: {reasoning}")
    except Exception as e:
        _add(score, 2, "simple_query_with_processing", False, f"invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Simple query with processing failed: {e}")


    # E) JOIN query tests (4 pts total)
    join_score = 0
    join_failures = []
    
    # E.1) Multi-table customer search (2 pts)
    try:
        state = TEST_STATES["e_1"]
        customers = [
            "Leonie Köhler",
            "Phil Hughes", 
            "Daan Peeters",
            "Kara Nielsen",
            "Alexandre Rocha",
            "Fernanda Ramos",
            "Jack Smith",
            "John Gordon",
            "Marc Dubois",
            "Ellie Sullivan",
            "Lucas Mancini",
            "Johannes Van der Berg",
            "Enrique Muñoz",
            "Emma Jones",
            "Diego Gutiérrez",
            "François Tremblay",
            "Madalena Sampaio",
            "Hannah Schneider",
            "Patrick Gray",
            "Julia Barnett",
            "Edward Francis",
            "Aaron Mitchell",
            "Wyatt Girard",
            "Frank Ralston",
            "Frank Harris",
            "Heather Leacock",
            "Martha Silk"
        ]
        out = app.invoke(state)
        with open("txt_dump/metallica_customers_test.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = all(customer in response for customer in customers)
        if ok:
            join_score += 2
        else:
            join_failures.append("Metallica customers test: response does not contain all expected customers")
    except Exception as e:
        join_failures.append(f"Metallica customers test failed: {e}")

    # E.2) Complex aggregation with JOINs (2 pts)
    try:
        state = TEST_STATES["e_2"]
        out = app.invoke(state)
        with open("txt_dump/top_sales_rep_test.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = "Jane Peacock" in response and "833.04" in response
        if ok:
            join_score += 2
        else:
            join_failures.append("Top sales rep test: response does not contain Jane Peacock and 833.04")
    except Exception as e:
        join_failures.append(f"Top sales rep test failed: {e}")

    # Add combined JOIN score
    _add(score, join_score, "join_query_tests", join_score != 0,
         f"Passed {join_score}/4 points" if join_score != 0 else f"All JOIN tests failed: {join_failures}")
    if join_failures:
        failures.extend(join_failures) 

    # F.1) Date range query test
    try:
        state = TEST_STATES["f_1"]
        employees = [
            "Andrew Adams",
            "Nancy Edwards", 
            "Jane Peacock",
            "Margaret Park"
        ]
        out = app.invoke(state)
        with open("txt_dump/date_range_query_test.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = all(employee in response for employee in employees)
        _add(score, 2, "date_range_query_test", ok,
             f"response: {response}" if ok else "response does not contain all employees")
        if not ok:
            failures.append("Response does not contain all employees")
    except Exception as e:
        _add(score, 2, "date_range_query_test", False, f"invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Date range query test failed: {e}")

    # F.2) Reject irrelevant queries
    try:
        state = TEST_STATES["f_2"]
        out = app.invoke(state)
        with open("txt_dump/reject_irrelevant_queries.txt", "w") as f:
            f.write(str(out))
        #response = out["messages"][-1].content
        response = _out_parser(out)
        ok = "don't know" in response.lower()
        _add(score, 2, "reject_irrelevant_queries", ok,
             f"response: {response}" if ok else f"response: {response}. It does not contain 'don't know'")
        if not ok:
            failures.append(f"Response does not contain 'don't know': {response}")
    except Exception as e:
        _add(score, 2, "reject_irrelevant_queries", False, f"invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Reject irrelevant queries test failed: {e}")

    _write_score(score)
    if failures:
        pytest.fail(" | ".join(failures))

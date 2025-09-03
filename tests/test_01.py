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
from test_utils.test_state import INITIAL_STATE, TEST_STATES, MINIMAL_STATE
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from test_utils.git_branch import get_git_branch

class LLMBinaryJudge(BaseModel):
    match: bool
    reasoning: str


CANDIDATE_NAME = get_git_branch()
DEFAULT_CONFIG_PATH = pathlib.Path.cwd() / "../langgraph.json"
DEFAULT_AGENT_FILENAME = os.getenv("DEFAULT_AGENT_FILENAME", "main.py")
DEFAULT_AGENT_PATH = pathlib.Path.cwd() / f"../{DEFAULT_AGENT_FILENAME}"

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

def company_object_parser(out):
    return dict(out["company_info"])

LLM_HOST_ALLOWLIST = [
    "api.openai.com",   
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "openai.azure.com",
    "api.cohere.ai", "cohere.ai",
]

TAVILY_HOST_ALLOWLIST = [
    "api.tavily.com",
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
    
    def tavily_calls(self):
        hits = []
        for url in self.urls:
            try:
                host = urllib.parse.urlparse(url).netloc
            except Exception:
                continue
            if any(h in host for h in TAVILY_HOST_ALLOWLIST):
                hits.append(url)
        return hits

@pytest.mark.asyncio
async def test_basics(monkeypatch):
    score = {"candidate": CANDIDATE_NAME, "bucket": "basic", "points": 0, "max_points": 22, "details": []}
    failures = []

    # A) Config validation (4 pts)
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

    # B) Real LLM call happens during invoke (2 pts)
    try:
        agent_path_env = os.getenv("CANDIDATE_AGENT_PATH", "").strip()
        agent_py_path = pathlib.Path(agent_path_env) if agent_path_env else DEFAULT_AGENT_PATH
        if not agent_py_path.exists():
            raise FileNotFoundError(f"agent.py not found at: {agent_py_path}")
        mod = _load_module(agent_py_path)
        app = _get_app(mod)
        spy = HttpSpy(monkeypatch)
        initial_state = INITIAL_STATE
        out = await app.ainvoke(initial_state)
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

    # C) Accepts minimal state (2 pts)
    try:
        state = MINIMAL_STATE
        out = await app.ainvoke(state)
        is_ok = isinstance(out, dict)
        _add(score, 2, "accepts_minimal_state", is_ok, "Accepts minimal state" if is_ok else "Does not accept minimal state")
        with open("txt_dump/accepts_minimal_state.txt", "w") as f:
            f.write(str(out))
    except Exception as e:
        _add(score, 2, "accepts_minimal_state", False, f"Invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Accepts minimal state failed: {e}")

    
    # D) company object has all the requested properties
    try:
        state = INITIAL_STATE
        out = await app.ainvoke(state)
        with open("txt_dump/company_object_has_all_properties.txt", "w") as f:
            f.write(str(out))
        response = company_object_parser(out)
        required_props = ["company_name", "founding_year", "founder_names", "product_description", "funding_summary", "notable_customers"]
        present_props = [prop for prop in required_props if prop in response]
        property_score = len(present_props)  # 1 point per property, max 6 (since we have 6 properties)
        if property_score == len(required_props):
            message = "Company object has all the requested properties: " + str(present_props)
        else:
            missing_props = [prop for prop in required_props if prop not in response]
            message = f"Company object missing properties: {', '.join(missing_props)} (has {len(present_props)}/{len(required_props)} properties)"
        
        _add(score, property_score, "company_object_has_all_properties", property_score > 0, message)
        if property_score < len(required_props):
            failures.append(message)
    except Exception as e:
        _add(score, 0, "company_object_has_all_properties", False, f"Invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Company object has all the requested properties failed: {e}")

    
    # E) all company properties hold information
    try:
        state = INITIAL_STATE
        out = await app.ainvoke(state)
        with open("txt_dump/all_company_properties_hold_information.txt", "w") as f:
            f.write(str(out))
        response = company_object_parser(out)
        required_props = ["company_name", "founding_year", "founder_names", "product_description", "funding_summary", "notable_customers"]
        
        # Check only the required properties that are present
        empty_props = []
        for prop in required_props:
            if prop in response and not response[prop]:
                empty_props.append(prop)
        
        properties_with_info = len([prop for prop in required_props if prop in response and response[prop]])
        
        if empty_props:
            message = f"Company object has empty properties: {', '.join(empty_props)} (properties with info: {properties_with_info}/{len(required_props)})"
        else:
            present_props = [prop for prop in required_props if prop in response]
            message = f"All present company properties have information (properties with info: {len(present_props)}/{len(required_props)})"

        _add(score, properties_with_info, "all_company_properties_hold_information", properties_with_info > 0, message)
        if empty_props:
            failures.append(message)
    except Exception as e:
        _add(score, 0, "all_company_properties_hold_information", False, f"Invoke failed: {type(e).__name__}: {e}")
        failures.append(f"All company properties hold information failed: {e}")


    # F) Request to Tavily API (2 pts)
    try:
        state = INITIAL_STATE
        out = await app.ainvoke(state)
        with open("txt_dump/tavily_api_call.txt", "w") as f:
            f.write(str(out))
        tavily_hits = spy.tavily_calls()
        with open("txt_dump/tavily_calls.txt", "w") as f:
            f.write(str(tavily_hits))
        ok = len(tavily_hits) > 0
        _add(score, 2, "tavily_api_call", ok,
             f"Tavily API calls detected: {len(tavily_hits)}" if ok else "No Tavily API calls detected during invoke")
        if not ok:
            failures.append("No outbound Tavily API calls observed during invoke.")
    except Exception as e:
        _add(score, 2, "tavily_api_call", False, f"invoke failed: {type(e).__name__}: {e}")
        failures.append(f"Tavily API call test failed: {e}")

    _write_score(score)
    if failures:
        pytest.fail(" | ".join(failures))

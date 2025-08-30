# smoke test
import importlib.util
import sys
import os
import json
import hashlib
import pathlib
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import subprocess
from test_utils.test_state import INITIAL_STATE
from test_utils.git_branch import get_git_branch



DEFAULT_AGENT_FILENAME = os.getenv("DEFAULT_AGENT_FILENAME", "main.py")
DEFAULT_AGENT_PATH = pathlib.Path.cwd() / f"../{DEFAULT_AGENT_FILENAME}"
CANDIDATE_NAME = get_git_branch()


def _load_module(agent_py_path: pathlib.Path):
    """Import agent.py as a temporary module; success == compiles/imports."""
    module_name = "candidate_agent_" + hashlib.md5(str(agent_py_path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(module_name, str(agent_py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    with open(agent_py_path, "r") as f:
        contents = f.read()
        with open("agent_test_file.txt", "w") as f:
            f.write(contents)
    try:
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, f"agent.py must compile/import: {e}"

def _get_app(mod):
    """Return global `app` from the module or raise a clear error."""
    if not hasattr(mod, "app") and not hasattr(mod, "agent"):
        raise AssertionError("agent.py must export a global variable `app` or `agent`")
    return mod.app or mod.agent

def _flatten_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            seg.get("text", "")
            for seg in content
            if isinstance(seg, dict) and seg.get("type") == "text"
        ).strip()
    return str(content).strip()

def _write_score(score):
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / f"smoke_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": (pts if ok else 0), "passed": bool(ok), "msg": msg})
    if ok:
        score["points"] += pts

# --- Smoke test (compile + basic invoke) ---
def test_smoke(monkeypatch):
    # Scoring model (10 pts total)
    score = {"candidate": CANDIDATE_NAME, "bucket": "smoke", "points": 0, "max_points": 10, "details": []}

    # A) Compile gate (2 pts)
    mod, err = _load_module(DEFAULT_AGENT_PATH)
    if mod is None:
        _add(score, 2, "compile", False, err)
        _write_score(score)
        pytest.skip(err)

    try:
        app = _get_app(mod)
        has_invoke = hasattr(app, "invoke")
        _add(score, 2, "compile", has_invoke, "compiled app must expose .invoke")
        if not has_invoke:
            _write_score(score)
            pytest.skip("Compiled app has no .invoke")
    except Exception as e:
        _add(score, 2, "compile", False, f"{type(e).__name__}: {e}")
        _write_score(score)
        pytest.skip(f"Compile failed: {e}")

    # B) Single invoke (8 pts, partial)
    try:
        initial_state = INITIAL_STATE
        out = app.invoke(initial_state)  # canonical initial state
        _add(score, 2, "invoke_accepts_canonical_state", True, "")
    except Exception as e:
        _add(score, 2, "invoke_accepts_canonical_state", False, f"{type(e).__name__}: {e}")
        _write_score(score)
        pytest.fail("Smoke invoke failed")

    ok_dict = isinstance(out, dict)
    with open("txt_dump/output.txt", "w") as f:
        f.write(str(out))
    _add(score, 1, "output_is_dict", ok_dict, "output must be a dict")
    if not ok_dict:
        _write_score(score)
        pytest.fail("Output is not a dict")

    msgs = out.get("messages", None)
    ok_msgs = isinstance(msgs, list) and len(msgs) > 0
    _add(score, 1, "messages_non_empty_list", ok_msgs, "messages must be a non-empty list")
    if not ok_msgs:
        _write_score(score)
        pytest.fail("messages must be a non-empty list")

    last_is_ai = isinstance(msgs[-1], AIMessage) or isinstance(msgs[-1], SystemMessage)
    txt_dump_dir = pathlib.Path("txt_dump")
    txt_dump_dir.mkdir(exist_ok=True, parents=True)
    with open("txt_dump/last_is_ai.txt", "w") as f:
        f.write(str(msgs))
    _add(score, 3, "last_is_ai", last_is_ai, "last message must be AIMessage")
    if not last_is_ai:
        _write_score(score)
        pytest.fail("Last message must be AIMessage")

    non_empty = bool(_flatten_text(msgs[-1].content))
    _add(score, 1, "last_ai_non_empty", non_empty, "final AI message does not have empty content")
    if not non_empty:
        _write_score(score)
        pytest.fail("Final AI message has empty content")
    _write_score(score)

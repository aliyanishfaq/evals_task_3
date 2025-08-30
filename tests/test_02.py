# graph distance
import os
import json
import pathlib
import pytest
import sys
from test_utils.git_branch import get_git_branch

from test_utils.graph_dist import compute_graph_distances


# Use git branch name as candidate name, with fallback to env var
CANDIDATE_NAME = get_git_branch()
EXPERT_SRC_PATH = pathlib.Path("expert_src").resolve()
CANDIDATE_AGENT_PATH = os.getenv("CANDIDATE_AGENT_PATH", "../main.py").strip()

def _write_score(score):
    out = pathlib.Path("results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"graph_dist_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": (pts if ok else 0), "passed": bool(ok), "msg": msg})
    if ok: 
        score["points"] += pts

def test_graph_distance():
    score = {"candidate": CANDIDATE_NAME, "bucket": "graph_dist", "points": 0, "max_points": 5, "details": []}
    
    try:
        # Check if candidate agent exists
        candidate_agent_file = pathlib.Path(CANDIDATE_AGENT_PATH)
        
        if not candidate_agent_file.exists():
            _add(score, 0, "graph_distance", False, f"No candidate agent found at {CANDIDATE_AGENT_PATH}")
            _write_score(score)
            pytest.skip(f"No candidate agent implementation found at {CANDIDATE_AGENT_PATH}")
        
        # Import candidate agent - handle different structures
        candidate_src_path = candidate_agent_file.parent
        sys.path.insert(0, str(candidate_src_path))
        
        # Clear any cached modules to ensure fresh import
        import importlib
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('agent') or 'simple_text2sql' in mod]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Try different import patterns
        if "agent/graph.py" in str(candidate_agent_file):
            from agent.graph import graph as candidate_graph
        else:
            # Assume it's like simple_text2sql.py with app export
            module_name = candidate_agent_file.stem
            mod = importlib.import_module(module_name)
            candidate_graph = mod.app
        
        # Remove candidate path and add expert path
        sys.path.remove(str(candidate_src_path))
        expert_src_path = str(EXPERT_SRC_PATH)
        sys.path.insert(0, expert_src_path)
        
        # Clear cached modules for expert import
        modules_to_remove = [mod for mod in sys.modules.keys() if 'simple_text2sql' in mod]
        for mod in modules_to_remove:
            del sys.modules[mod]
            
        from simple_text2sql import app as gold_graph

        # draw gold graph
        with open("gold_graph.png", "wb") as f:
            f.write(gold_graph.get_graph().draw_mermaid_png())
        
        # Generate graph visualization
        with open("candidate_graph.png", "wb") as f:
            f.write(candidate_graph.get_graph().draw_mermaid_png())
        
        # Calculate edit distance
        distance = compute_graph_distances(candidate_graph, gold_graph)
        print(f"Structural edit distance: {distance}")
        
        # Calculate score based on the formula
        MAX_PTS = 5
        D_CAP = 10
        
        points = max(0, round(MAX_PTS * (1 - min(distance, D_CAP) / D_CAP)))
        print(f"Score: {points}/{MAX_PTS} points")
        
        _add(score, points, "graph_distance", True, f"Edit distance: {distance}")
        
    except Exception as e:
        _add(score, 0, "graph_distance", False, f"Error: {type(e).__name__}: {e}")
        pytest.fail(f"Graph distance test failed: {e}")
    
    _write_score(score)

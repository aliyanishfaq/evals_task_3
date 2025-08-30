"""
Graph Distance Comparison for LangGraphs

This module provides utilities to compare LangGraph workflows using
various graph distance algorithms and similarity metrics.
"""

import networkx as nx
from typing import Dict, Any, Tuple
import numpy as np

def langgraph_to_networkx(app) -> nx.DiGraph:
    """Convert LangGraph to NetworkX directed graph"""
    G = nx.DiGraph()
    
    # Get graph information
    graph_info = app.get_graph()
    
    # Add nodes with attributes
    for node_id, node_data in graph_info.nodes.items():
        node_type = type(node_data.data).__name__ if node_data.data else "None"
        #id='__start__', name='__start__', data=RunnableCallable(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None
        print(f"Node {node_id} has type {node_type}")
        G.add_node(
            node_id, 
            name=node_data.name,
            is_start=node_id == "__start__",
            is_end=node_id == "__end__"
        )
    
    # Add edges with attributes
    for edge in graph_info.edges:
        G.add_edge(
            edge.source, 
            edge.target, 
            conditional=edge.conditional,
        )
    
    return G

def compute_graph_distances(app1, app2) -> float:
    """Compute various distance metrics between two LangGraphs"""
    G1 = langgraph_to_networkx(app1)
    G2 = langgraph_to_networkx(app2)
    
    results = {}
    
    # 1. Graph Edit Distance (if graphs are small enough)
    try:
        edit_dist = nx.graph_edit_distance(G1, G2, timeout=60)
        print(f"Edit distance: {edit_dist}")
        results['edit_distance'] = edit_dist
    except Exception as e:
        print(f"Edit distance failed: {e}")
        results['edit_distance'] = float('inf')
    
    return results['edit_distance']
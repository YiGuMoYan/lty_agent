import os
import json
import pickle
import time
import networkx as nx
from config import GRAPH_PATH, PICKLE_PATH, DEBUG

def load_graph():
    """
    Loads the Knowledge Graph slightly prioritized from Pickle (Fast) then JSON (Slow).
    Returns: nx.Graph or None
    """
    graph = None
    t0 = time.time()
    
    # 1. Try Pickle
    if os.path.exists(PICKLE_PATH):
        try:
            with open(PICKLE_PATH, 'rb') as f:
                graph = pickle.load(f)
            if DEBUG: print(f"  [System] Fast-loaded graph from binary cache in {time.time()-t0:.4f}s")
            return graph
        except Exception as e:
            print(f"  [System] Binary cache load failed: {e}")

    # 2. Try JSON
    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
                graph = nx.node_link_graph(json.load(f))
            if DEBUG: print(f"  [System] Loaded graph from JSON source in {time.time()-t0:.4f}s")
            return graph
        except Exception as e:
            print(f"  [System] JSON load failed: {e}")
            
    return None

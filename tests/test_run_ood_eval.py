import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from run_ood_eval import _weight_sensitive

class DummyGraph:
    def __init__(self, edges, s, t):
        self.edges = edges
        self.source = s
        self.target = t
        self.node_ids = set()
        for u, v, _ in edges:
            self.node_ids.add(u)
            self.node_ids.add(v)
            
    def adjacency(self):
        adj = {n: [] for n in self.node_ids}
        for u, v, w in self.edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        return adj

class DummySP:
    def __init__(self, path):
        self.path = path

def test_weight_sensitive():
    # Graph: 0-1-2-3 (w=1 each) and 0-3 (w=10)
    # BFS min hop is 1 (0-3). 
    g = DummyGraph([(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 10)], 0, 3)
    
    # Dijkstra path is 0-1-2-3, hop count 3. 
    # 3 != 1 => True
    sp_sensitive = DummySP([0, 1, 2, 3])
    assert _weight_sensitive(g, sp_sensitive) == True

    # Same graph but assume SP is 0-3 (hop count 1)
    sp_insensitive = DummySP([0, 3])
    assert _weight_sensitive(g, sp_insensitive) == False

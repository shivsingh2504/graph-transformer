
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data.graph_generator import Graph


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortestPath:
    path: List[int]
    cost: int


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def run_dijkstra(graph: Graph) -> ShortestPath:
    source: int = graph.source
    target: int = graph.target

    # ------------------------------------------------------------------
    # Trivial case
    # ------------------------------------------------------------------
    if source == target:
        return ShortestPath(path=[source], cost=0)

    # ------------------------------------------------------------------
    # Initialise data structures
    # ------------------------------------------------------------------
    adj: Dict[int, List[Tuple[int, int]]] = graph.adjacency()

    # dist[v] = best known cumulative cost from source to v.
    dist: Dict[int, float] = {node: math.inf for node in adj}
    dist[source] = 0  # <-- THIS LINE: must be `dist`, not `dict`

    # prev[v] = predecessor of v on the current best path from source.
    prev: Dict[int, Optional[int]] = {node: None for node in adj}

    # Min-heap entries: (tentative_cost, node).
    heap: List[Tuple[float, int]] = [(0, source)]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while heap:
        cost_u, u = heapq.heappop(heap)

        # Lazy deletion: skip if a shorter path to u was already finalised.
        if cost_u > dist[u]:
            continue

        # Early exit: once the target is popped its distance is finalised.
        if u == target:
            break

        for v, weight in adj[u]:
            alt: float = dist[u] + weight
            if alt < dist[v]:          # strict <  →  first relaxation wins on ties
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    # ------------------------------------------------------------------
    # Connectivity guard
    # ------------------------------------------------------------------
    if dist[target] == math.inf:
        raise ValueError(
            f"Target node {target} is unreachable from source {source}.  "
            f"This should not occur for graphs produced by the generator, "
            f"which guarantees connectivity."
        )

    # ------------------------------------------------------------------
    # Path reconstruction (walk prev[] backwards from target)
    # ------------------------------------------------------------------
    path: List[int] = []
    cursor: Optional[int] = target
    while cursor is not None:
        path.append(cursor)
        cursor = prev[cursor]
    path.reverse()

    return ShortestPath(path=path, cost=int(dist[target]))


# ---------------------------------------------------------------------------
# Minimal demo (mirrors graph_generator.py __main__ style)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # dijkstra.py lives at src/data/dijkstra.py.
    # One ".." reaches src/, which is the package root that "from data.X import"
    # requires.  Two levels would overshoot to the project root.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.graph_generator import generate_random_connected_graph

    g = generate_random_connected_graph(num_nodes=8, seed=42, edge_density=0.4)
    result = run_dijkstra(g)
    print("=== Dijkstra Demo ===")
    print(f"Nodes  : {g.num_nodes}")
    print(f"Source : {g.source}  |  Target : {g.target}")
    print(f"Path   : {result.path}")
    print(f"Cost   : {result.cost}")

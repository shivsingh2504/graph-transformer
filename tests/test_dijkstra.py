
from __future__ import annotations

import os
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from data.graph_generator import Graph, generate_random_connected_graph
from data.dijkstra import ShortestPath, run_dijkstra


def _make_graph(
    num_nodes: int,
    edges: List[Tuple[int, int, int]],
    source: int,
    target: int,
    seed: int = 0,
) -> Graph:
    return Graph(
        num_nodes=num_nodes,
        edges=edges,
        source=source,
        target=target,
        seed=seed,
    )


def _brute_force_min_cost(
    adj: Dict[int, List[Tuple[int, int]]],
    source: int,
    target: int,
) -> int:
    best: List[float] = [float("inf")]

    def _dfs(node: int, cost: int, visited: Set[int]) -> None:
        if node == target:
            if cost < best[0]:
                best[0] = cost
            return
        for neighbour, weight in adj[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                _dfs(neighbour, cost + weight, visited)
                visited.remove(neighbour)

    _dfs(source, 0, {source})
    return int(best[0])


# ===========================================================================
# 1. Hand-crafted graphs with known shortest paths
# ===========================================================================

class TestHandCrafted:
    def test_triangle_takes_two_hop_path(self) -> None:
        g = _make_graph(
            num_nodes=3,
            edges=[(0, 1, 4), (0, 2, 10), (1, 2, 3)],
            source=0,
            target=2,
        )
        result = run_dijkstra(g)
        assert result.cost == 7
        assert result.path == [0, 1, 2]

    def test_linear_chain(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 1), (1, 2, 1), (2, 3, 1)],
            source=0,
            target=3,
        )
        result = run_dijkstra(g)
        assert result.cost == 3
        assert result.path == [0, 1, 2, 3]

    def test_diamond_prefers_cheaper_arm(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 1), (0, 2, 4), (1, 3, 2), (2, 3, 1)],
            source=0,
            target=3,
        )
        result = run_dijkstra(g)
        assert result.cost == 3
        assert result.path == [0, 1, 3]

    def test_diamond_reversed_direction(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 1), (0, 2, 4), (1, 3, 2), (2, 3, 1)],
            source=3,
            target=0,
        )
        result = run_dijkstra(g)
        assert result.cost == 3
        assert result.path == [3, 1, 0]

    def test_five_node_detour(self) -> None:
        g = _make_graph(
            num_nodes=5,
            edges=[
                (0, 1, 10),
                (0, 2, 3),
                (1, 2, 4),
                (1, 3, 2),
                (2, 3, 8),
                (3, 4, 5),
            ],
            source=0,
            target=4,
        )
        result = run_dijkstra(g)
        assert result.cost == 14
        assert result.path == [0, 2, 1, 3, 4]

    def test_single_edge_graph(self) -> None:
        g = _make_graph(
            num_nodes=2,
            edges=[(0, 1, 7)],
            source=0,
            target=1,
        )
        result = run_dijkstra(g)
        assert result.cost == 7
        assert result.path == [0, 1]

    def test_high_weight_bypass(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 100), (0, 2, 1), (1, 3, 1), (2, 3, 1)],
            source=0,
            target=1,
        )
        result = run_dijkstra(g)
        assert result.cost == 3
        assert result.path == [0, 2, 3, 1]


# ===========================================================================
# 2. Tie-breaking behaviour
# ===========================================================================

class TestTieBreaking:
    def test_equal_cost_paths_first_settled_wins(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 2), (0, 2, 1), (1, 3, 1), (2, 3, 2)],
            source=0,
            target=3,
        )
        result = run_dijkstra(g)
        assert result.cost == 3          # both paths are optimal
        assert result.path == [0, 2, 3]  # first-settled wins: node 2 before node 1

    def test_tie_breaking_is_stable_across_repeated_calls(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 2), (0, 2, 1), (1, 3, 1), (2, 3, 2)],
            source=0,
            target=3,
        )
        r1 = run_dijkstra(g)
        r2 = run_dijkstra(g)
        assert r1.path == r2.path
        assert r1.cost == r2.cost


# ===========================================================================
# 3. source == target edge case
# ===========================================================================

class TestSourceEqualsTarget:

    def test_source_equals_target_returns_singleton_path(self) -> None:
        g = _make_graph(
            num_nodes=4,
            edges=[(0, 1, 5), (1, 2, 3), (0, 3, 2), (2, 3, 4)],
            source=2,
            target=2,
        )
        result = run_dijkstra(g)
        assert result.path == [2]
        assert result.cost == 0

    @pytest.mark.parametrize("node", [0, 1, 5, 9])
    def test_source_equals_target_parametrized(self, node: int) -> None:
        base = generate_random_connected_graph(10, seed=42)
        g = Graph(
            num_nodes=base.num_nodes,
            edges=base.edges,
            source=node,
            target=node,
            seed=base.seed,
        )
        result = run_dijkstra(g)
        assert result.path == [node]
        assert result.cost == 0


# ===========================================================================
# 4. Endpoint correctness
# ===========================================================================

class TestEndpoints:

    @pytest.mark.parametrize("seed", range(20))
    def test_path_starts_at_source(self, seed: int) -> None:
        g = generate_random_connected_graph(12, seed, edge_density=0.4)
        assert run_dijkstra(g).path[0] == g.source

    @pytest.mark.parametrize("seed", range(20))
    def test_path_ends_at_target(self, seed: int) -> None:
        g = generate_random_connected_graph(12, seed, edge_density=0.4)
        assert run_dijkstra(g).path[-1] == g.target

    @pytest.mark.parametrize("seed", range(10))
    def test_path_has_at_least_two_nodes_when_source_ne_target(
        self, seed: int
    ) -> None:
        g = generate_random_connected_graph(10, seed)
        assert g.source != g.target  # guaranteed by generator
        assert len(run_dijkstra(g).path) >= 2


# ===========================================================================
# 5. Path edge validity
# ===========================================================================

class TestPathValidity:

    @pytest.mark.parametrize("seed", range(25))
    def test_consecutive_pairs_are_real_edges_density_mode(
        self, seed: int
    ) -> None:
        g = generate_random_connected_graph(15, seed, edge_density=0.4)
        self._assert_path_edges_valid(g)

    @pytest.mark.parametrize("seed", range(15))
    def test_consecutive_pairs_are_real_edges_num_edges_mode(
        self, seed: int
    ) -> None:
        # 15 nodes → spanning tree needs 14 edges; request 25 for moderate density.
        g = generate_random_connected_graph(15, seed, num_edges=25)
        self._assert_path_edges_valid(g)

    @pytest.mark.parametrize("seed", range(10))
    def test_consecutive_pairs_are_real_edges_spanning_tree(
        self, seed: int
    ) -> None:
        g = generate_random_connected_graph(12, seed)
        self._assert_path_edges_valid(g)

    @staticmethod
    def _assert_path_edges_valid(graph: Graph) -> None:
        result = run_dijkstra(graph)
        adj = graph.adjacency()
        neighbours: Dict[int, Set[int]] = {
            node: {nb for nb, _ in nbs} for node, nbs in adj.items()
        }
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            assert v in neighbours[u], (
                f"Edge ({u},{v}) in returned path is not present in the graph."
            )

    @pytest.mark.parametrize("seed", range(15))
    def test_path_nodes_in_range(self, seed: int) -> None:
        g = generate_random_connected_graph(12, seed, edge_density=0.5)
        for node in run_dijkstra(g).path:
            assert node in g.node_ids

    @pytest.mark.parametrize("seed", range(15))
    def test_path_is_simple_no_repeated_nodes(self, seed: int) -> None:
        g = generate_random_connected_graph(12, seed, edge_density=0.5)
        path = run_dijkstra(g).path
        assert len(path) == len(set(path)), (
            "Path contains repeated nodes — not a simple path."
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_cost_matches_sum_of_edge_weights_in_path(self, seed: int) -> None:
        g = generate_random_connected_graph(10, seed, edge_density=0.5)
        result = run_dijkstra(g)
        adj = g.adjacency()
        weight_lookup: Dict[Tuple[int, int], int] = {
            (node, nb): w
            for node, neighbours in adj.items()
            for nb, w in neighbours
        }
        computed_cost = sum(
            weight_lookup[(result.path[i], result.path[i + 1])]
            for i in range(len(result.path) - 1)
        )
        assert computed_cost == result.cost


# ===========================================================================
# 6. Determinism
# ===========================================================================

class TestDeterminism:

    @pytest.mark.parametrize(
        "num_nodes, seed",
        [(5, 0), (10, 7), (20, 42), (50, 999)],
    )
    def test_repeated_calls_identical(self, num_nodes: int, seed: int) -> None:
        g = generate_random_connected_graph(num_nodes, seed, edge_density=0.4)
        r1 = run_dijkstra(g)
        r2 = run_dijkstra(g)
        assert r1.path == r2.path
        assert r1.cost == r2.cost


# ===========================================================================
# 7. Optimality – brute-force cross-check (small graphs, stdlib only)
# ===========================================================================

class TestOptimalityBruteForce:
    # num_nodes=8, edge_density=0.5 → ~14 edges.  DFS is tractable.
    @pytest.mark.parametrize("seed", range(30))  # keep num_nodes <= 8 here
    def test_cost_is_minimum_over_all_paths_density_mode(
        self, seed: int
    ) -> None:
        g = generate_random_connected_graph(8, seed, edge_density=0.5)
        result = run_dijkstra(g)
        expected = _brute_force_min_cost(g.adjacency(), g.source, g.target)
        assert result.cost == expected, (
            f"seed={seed}: Dijkstra cost {result.cost}, brute-force {expected}."
        )

    # num_nodes=8 only — see size constraint note above.
    @pytest.mark.parametrize("seed", range(20))  # keep num_nodes <= 8 here
    def test_cost_is_minimum_spanning_tree_graph(self, seed: int) -> None:
        g = generate_random_connected_graph(8, seed)
        result = run_dijkstra(g)
        expected = _brute_force_min_cost(g.adjacency(), g.source, g.target)
        assert result.cost == expected

    # num_nodes=8 only — see size constraint note above.
    @pytest.mark.parametrize("seed", range(15))  # keep num_nodes <= 8 here
    def test_cost_is_minimum_num_edges_mode(self, seed: int) -> None:
        # 8 nodes → spanning tree needs 7 edges; 14 = half of max(28).
        g = generate_random_connected_graph(8, seed, num_edges=14)
        result = run_dijkstra(g)
        expected = _brute_force_min_cost(g.adjacency(), g.source, g.target)
        assert result.cost == expected, (
            f"seed={seed}: Dijkstra cost {result.cost}, brute-force {expected}."
        )

    def test_hand_crafted_triangle_brute_force_agrees(self) -> None:
        g = _make_graph(
            num_nodes=3,
            edges=[(0, 1, 4), (0, 2, 10), (1, 2, 3)],
            source=0,
            target=2,
        )
        assert run_dijkstra(g).cost == _brute_force_min_cost(
            g.adjacency(), g.source, g.target
        )

    def test_hand_crafted_five_node_brute_force_agrees(self) -> None:
        g = _make_graph(
            num_nodes=5,
            edges=[
                (0, 1, 10),
                (0, 2, 3),
                (1, 2, 4),
                (1, 3, 2),
                (2, 3, 8),
                (3, 4, 5),
            ],
            source=0,
            target=4,
        )
        assert run_dijkstra(g).cost == _brute_force_min_cost(
            g.adjacency(), g.source, g.target
        )


# ===========================================================================
# 8. Optimality – NetworkX cross-check (skipped if not installed)
# ===========================================================================

class TestNetworkX:
    @pytest.fixture(autouse=True)
    def _require_networkx(self):
        pytest.importorskip("networkx")

    def _nx_shortest(self, graph: Graph) -> Tuple[int, List[int]]:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        for u, v, w in graph.edges:
            G.add_edge(u, v, weight=w)
        cost, path = nx.single_source_dijkstra(
            G, source=graph.source, target=graph.target, weight="weight"
        )
        return int(cost), path

    @pytest.mark.parametrize("seed", range(30))
    def test_cost_matches_networkx_density_mode(self, seed: int) -> None:
        g = generate_random_connected_graph(20, seed, edge_density=0.4)
        nx_cost, _ = self._nx_shortest(g)
        assert run_dijkstra(g).cost == nx_cost, (
            f"seed={seed}: Dijkstra vs NetworkX cost mismatch."
        )

    @pytest.mark.parametrize("seed", range(15))
    def test_cost_matches_networkx_num_edges_mode(self, seed: int) -> None:
        """Graphs built via num_edges= cross-checked against NetworkX."""
        # 20 nodes → spanning tree 19 edges; request 50 for moderate density.
        g = generate_random_connected_graph(20, seed, num_edges=50)
        nx_cost, _ = self._nx_shortest(g)
        assert run_dijkstra(g).cost == nx_cost

    @pytest.mark.parametrize("seed", range(15))
    def test_cost_matches_networkx_dense_graph(self, seed: int) -> None:
        g = generate_random_connected_graph(15, seed, edge_density=0.8)
        nx_cost, _ = self._nx_shortest(g)
        assert run_dijkstra(g).cost == nx_cost

    @pytest.mark.parametrize("seed", range(15))
    def test_spanning_tree_path_matches_networkx_exactly(
        self, seed: int
    ) -> None:
        """On a spanning tree there is exactly one simple path — paths must match."""
        g = generate_random_connected_graph(12, seed)
        nx_cost, nx_path = self._nx_shortest(g)
        result = run_dijkstra(g)
        assert result.cost == nx_cost
        assert result.path == nx_path

    @pytest.mark.parametrize("seed", range(10))
    def test_complete_graph_cost_matches_networkx(self, seed: int) -> None:
        """Complete graph (edge_density=1.0): every edge present."""
        g = generate_random_connected_graph(8, seed, edge_density=1.0)
        nx_cost, _ = self._nx_shortest(g)
        assert run_dijkstra(g).cost == nx_cost

    @pytest.mark.parametrize(
        "num_nodes, seed",
        [(10, 0), (20, 1), (30, 2), (50, 3)],
    )
    def test_scales_to_larger_graphs(self, num_nodes: int, seed: int) -> None:
        g = generate_random_connected_graph(num_nodes, seed, edge_density=0.3)
        nx_cost, _ = self._nx_shortest(g)
        assert run_dijkstra(g).cost == nx_cost


# ===========================================================================
# 9. ShortestPath dataclass
# ===========================================================================

class TestShortestPathDataclass:
    """Verify the return type's fields and invariants."""

    def test_has_path_and_cost_fields(self) -> None:
        g = generate_random_connected_graph(8, seed=0)
        result = run_dijkstra(g)
        assert hasattr(result, "path")
        assert hasattr(result, "cost")

    def test_path_is_list_of_ints(self) -> None:
        g = generate_random_connected_graph(8, seed=0)
        result = run_dijkstra(g)
        assert isinstance(result.path, list)
        assert all(isinstance(n, int) for n in result.path)

    def test_cost_is_int(self) -> None:
        g = generate_random_connected_graph(8, seed=0)
        result = run_dijkstra(g)
        assert isinstance(result.cost, int)

    def test_cost_is_non_negative(self) -> None:
        g = generate_random_connected_graph(8, seed=0)
        assert run_dijkstra(g).cost >= 0

    def test_frozen_path_field_not_reassignable(self) -> None:
        """ShortestPath is frozen — field reassignment must raise."""
        result = run_dijkstra(generate_random_connected_graph(6, seed=0))
        with pytest.raises((AttributeError, TypeError)):
            result.path = []  # type: ignore[misc]

    def test_frozen_cost_field_not_reassignable(self) -> None:
        result = run_dijkstra(generate_random_connected_graph(6, seed=0))
        with pytest.raises((AttributeError, TypeError)):
            result.cost = -1  # type: ignore[misc]

from __future__ import annotations

import os
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from data.graph_generator import Graph, generate_random_connected_graph
from data.dijkstra import ShortestPath, run_dijkstra
from data.dataset_generator import DatasetSplit, generate_dataset_split

def _path_edges_valid(graph: Graph, path: List[int]) -> bool:
    adj = graph.adjacency()
    neighbours: Dict[int, Set[int]] = {
        node: {nb for nb, _ in nbs} for node, nbs in adj.items()
    }
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if v not in neighbours[u]:
            return False
    return True


def _cost_matches_path(graph: Graph, path: List[int], cost: int) -> bool:
    adj = graph.adjacency()
    weight_lookup: Dict[Tuple[int, int], int] = {
        (node, nb): w for node, nbs in adj.items() for nb, w in nbs
    }
    computed = sum(
        weight_lookup[(path[i], path[i + 1])] for i in range(len(path) - 1)
    )
    return computed == cost


class TestExactCount:
    @pytest.mark.parametrize("num_examples", [0, 1, 5, 50, 200])
    def test_len_matches_requested(self, num_examples: int) -> None:
        split = generate_dataset_split(
            num_examples=num_examples, node_range=(5, 10), base_seed=0
        )
        assert len(split.examples) == num_examples

    @pytest.mark.parametrize("num_examples", [0, 1, 5, 50, 200])
    def test_num_examples_field_matches(self, num_examples: int) -> None:
        split = generate_dataset_split(
            num_examples=num_examples, node_range=(5, 10), base_seed=0
        )
        assert split.num_examples == num_examples

    @pytest.mark.parametrize("num_examples", [0, 1, 5, 50, 200])
    def test_num_examples_field_equals_len(self, num_examples: int) -> None:
        split = generate_dataset_split(
            num_examples=num_examples, node_range=(5, 10), base_seed=0
        )
        assert split.num_examples == len(split.examples)


class TestDeterminism:
    @pytest.mark.parametrize(
        "num_examples, node_range, base_seed",
        [
            (10, (5, 10), 0),
            (10, (5, 20), 42),
            (5, (25, 50), 999),
            (1, (5, 5), 7),
        ],
    )
    def test_repeated_call_identical(
        self, num_examples: int, node_range: Tuple[int, int], base_seed: int
    ) -> None:
        kwargs = dict(
            num_examples=num_examples,
            node_range=node_range,
            base_seed=base_seed,
            edge_density=0.4,
        )
        split1 = generate_dataset_split(**kwargs)
        split2 = generate_dataset_split(**kwargs)
        assert len(split1.examples) == len(split2.examples)
        for (g1, sp1), (g2, sp2) in zip(split1.examples, split2.examples):
            assert g1 == g2
            assert sp1 == sp2

    def test_order_is_deterministic(self) -> None:
        split1 = generate_dataset_split(num_examples=20, node_range=(5, 15), base_seed=3)
        split2 = generate_dataset_split(num_examples=20, node_range=(5, 15), base_seed=3)
        for (g1, _), (g2, _) in zip(split1.examples, split2.examples):
            assert g1.seed == g2.seed
            assert g1.num_nodes == g2.num_nodes

    def test_determinism_independent_of_prior_random_state(self) -> None:
        import random as _random

        split_before = generate_dataset_split(
            num_examples=10, node_range=(5, 10), base_seed=0
        )
        _random.seed(999_999)
        split_after = generate_dataset_split(
            num_examples=10, node_range=(5, 10), base_seed=0
        )
        for (g1, sp1), (g2, sp2) in zip(split_before.examples, split_after.examples):
            assert g1 == g2
            assert sp1 == sp2


class TestDifferentSeeds:
    @pytest.mark.parametrize("seed_a, seed_b", [(0, 1), (42, 43), (0, 100)])
    def test_different_seeds_produce_different_graphs(
        self, seed_a: int, seed_b: int
    ) -> None:
        split_a = generate_dataset_split(num_examples=20, node_range=(5, 20), base_seed=seed_a)
        split_b = generate_dataset_split(num_examples=20, node_range=(5, 20), base_seed=seed_b)
        graphs_a = [g.to_dict() for g, _ in split_a.examples]
        graphs_b = [g.to_dict() for g, _ in split_b.examples]
        assert graphs_a != graphs_b

    def test_different_seeds_produce_different_paths(self) -> None:
        split_0 = generate_dataset_split(num_examples=20, node_range=(8, 15), base_seed=0)
        split_1 = generate_dataset_split(num_examples=20, node_range=(8, 15), base_seed=1)
        assert [sp.path for _, sp in split_0.examples] != [sp.path for _, sp in split_1.examples]


class TestNodeRange:
    @pytest.mark.parametrize(
        "node_range", [(5, 20), (2, 2), (10, 10), (25, 50), (2, 5)]
    )
    def test_all_nodes_within_range(self, node_range: Tuple[int, int]) -> None:
        min_nodes, max_nodes = node_range
        split = generate_dataset_split(num_examples=30, node_range=node_range, base_seed=77)
        for graph, _ in split.examples:
            assert min_nodes <= graph.num_nodes <= max_nodes

    def test_fixed_node_count_when_min_equals_max(self) -> None:
        n = 7
        split = generate_dataset_split(num_examples=20, node_range=(n, n), base_seed=0)
        for graph, _ in split.examples:
            assert graph.num_nodes == n

    def test_node_counts_vary_when_range_is_wide(self) -> None:
        split = generate_dataset_split(num_examples=50, node_range=(5, 20), base_seed=0)
        assert len({g.num_nodes for g, _ in split.examples}) > 1


class TestPairConsistency:
    @pytest.fixture
    def sample_split(self) -> DatasetSplit:
        return generate_dataset_split(
            num_examples=30, node_range=(5, 20), base_seed=42, edge_density=0.4
        )

    def test_path_starts_at_source(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert sp.path[0] == graph.source

    def test_path_ends_at_target(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert sp.path[-1] == graph.target

    def test_path_nodes_in_valid_range(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            for node in sp.path:
                assert 0 <= node < graph.num_nodes

    def test_path_is_simple(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert len(sp.path) == len(set(sp.path))

    def test_path_has_at_least_two_nodes(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert len(sp.path) >= 2

    def test_consecutive_pairs_are_real_edges(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert _path_edges_valid(graph, sp.path)

    def test_cost_equals_sum_of_edge_weights(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert _cost_matches_path(graph, sp.path, sp.cost)

    def test_cost_is_non_negative(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert sp.cost >= 0

    def test_cost_is_int(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert isinstance(sp.cost, int)

    def test_path_is_list_of_ints(self, sample_split: DatasetSplit) -> None:
        for graph, sp in sample_split.examples:
            assert isinstance(sp.path, list)
            assert all(isinstance(n, int) for n in sp.path)

    def test_graph_source_differs_from_target(self, sample_split: DatasetSplit) -> None:
        for graph, _ in sample_split.examples:
            assert graph.source != graph.target

    def test_shortest_path_is_frozen(self) -> None:
        split = generate_dataset_split(num_examples=1, node_range=(5, 10), base_seed=0)
        _, sp = split.examples[0]
        with pytest.raises((AttributeError, TypeError)):
            sp.path = []  # type: ignore[misc]

    def test_dijkstra_cross_check(self) -> None:
        split = generate_dataset_split(
            num_examples=20, node_range=(5, 15), base_seed=0, edge_density=0.4
        )
        for graph, sp in split.examples:
            recomputed = run_dijkstra(graph)
            assert recomputed.path == sp.path
            assert recomputed.cost == sp.cost


class TestEdgeCases:
    def test_zero_examples_returns_empty_split(self) -> None:
        split = generate_dataset_split(num_examples=0, node_range=(5, 20), base_seed=0)
        assert split.examples == []
        assert split.num_examples == 0

    def test_zero_examples_is_deterministic(self) -> None:
        split1 = generate_dataset_split(num_examples=0, node_range=(5, 20), base_seed=0)
        split2 = generate_dataset_split(num_examples=0, node_range=(5, 20), base_seed=0)
        assert split1.examples == split2.examples

    def test_one_example_returns_single_element_list(self) -> None:
        split = generate_dataset_split(num_examples=1, node_range=(5, 10), base_seed=0)
        assert len(split.examples) == 1

    def test_one_example_pair_is_valid(self) -> None:
        split = generate_dataset_split(num_examples=1, node_range=(5, 10), base_seed=0)
        graph, sp = split.examples[0]
        assert sp.path[0] == graph.source
        assert sp.path[-1] == graph.target
        assert _path_edges_valid(graph, sp.path)
        assert _cost_matches_path(graph, sp.path, sp.cost)

    def test_fixed_node_count_single_node_value(self) -> None:
        split = generate_dataset_split(num_examples=10, node_range=(5, 5), base_seed=0)
        for graph, _ in split.examples:
            assert graph.num_nodes == 5

    def test_minimum_valid_node_range(self) -> None:
        split = generate_dataset_split(num_examples=5, node_range=(2, 2), base_seed=0)
        for graph, sp in split.examples:
            assert graph.num_nodes == 2
            assert len(sp.path) >= 2

    def test_large_split_has_correct_length(self) -> None:
        split = generate_dataset_split(num_examples=500, node_range=(5, 20), base_seed=0)
        assert len(split.examples) == 500


class TestInvalidArgs:
    def test_negative_num_examples_raises(self) -> None:
        with pytest.raises(ValueError, match="num_examples"):
            generate_dataset_split(num_examples=-1, node_range=(5, 10), base_seed=0)

    def test_min_nodes_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="min_nodes"):
            generate_dataset_split(num_examples=5, node_range=(1, 10), base_seed=0)

    def test_min_nodes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_nodes"):
            generate_dataset_split(num_examples=5, node_range=(0, 10), base_seed=0)

    def test_max_nodes_less_than_min_nodes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_nodes"):
            generate_dataset_split(num_examples=5, node_range=(10, 5), base_seed=0)

    def test_invalid_min_weight_propagates(self) -> None:
        with pytest.raises(ValueError, match="min_weight"):
            generate_dataset_split(
                num_examples=1, node_range=(5, 10), base_seed=0, min_weight=0
            )

    def test_invalid_edge_density_propagates(self) -> None:
        with pytest.raises(ValueError, match="edge_density"):
            generate_dataset_split(
                num_examples=1, node_range=(5, 10), base_seed=0, edge_density=0.0
            )

    def test_num_edges_too_large_for_min_nodes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_edges"):
            generate_dataset_split(
                num_examples=10, node_range=(5, 20), base_seed=0, num_edges=15
            )

    def test_num_edges_too_large_for_min_nodes_raises_at_boundary(self) -> None:
        min_nodes = 6
        max_edges_for_min = min_nodes * (min_nodes - 1) // 2
        with pytest.raises(ValueError, match="num_edges"):
            generate_dataset_split(
                num_examples=5,
                node_range=(min_nodes, 20),
                base_seed=0,
                num_edges=max_edges_for_min + 1,
            )

    def test_num_edges_at_max_for_min_nodes_is_accepted(self) -> None:
        min_nodes = 6
        max_edges_for_min = min_nodes * (min_nodes - 1) // 2
        split = generate_dataset_split(
            num_examples=5,
            node_range=(min_nodes, min_nodes),
            base_seed=0,
            num_edges=max_edges_for_min,
        )
        assert split.num_examples == 5
        for graph, _ in split.examples:
            assert len(graph.edges) == max_edges_for_min

    def test_seed_collision_guard_raises_on_excessive_num_examples(self) -> None:
        from data.dataset_generator import _SEED_MULTIPLIER
        with pytest.raises(ValueError, match="num_examples"):
            generate_dataset_split(
                num_examples=_SEED_MULTIPLIER + 1, node_range=(5, 10), base_seed=0
            )

    def test_seed_collision_guard_raises_on_out_of_range_base_seed(self) -> None:
        from data.dataset_generator import _SEED_MULTIPLIER
        with pytest.raises(ValueError, match="base_seed"):
            generate_dataset_split(
                num_examples=5, node_range=(5, 10), base_seed=_SEED_MULTIPLIER
            )

    def test_negative_base_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="base_seed"):
            generate_dataset_split(num_examples=5, node_range=(5, 10), base_seed=-1)


class TestMetadata:
    def test_node_range_stored(self) -> None:
        split = generate_dataset_split(num_examples=5, node_range=(7, 14), base_seed=0)
        assert split.node_range == (7, 14)

    def test_base_seed_stored(self) -> None:
        split = generate_dataset_split(num_examples=5, node_range=(5, 10), base_seed=123)
        assert split.base_seed == 123

    def test_num_examples_field_always_equals_len(self) -> None:
        for n in [0, 1, 10, 100]:
            split = generate_dataset_split(num_examples=n, node_range=(5, 10), base_seed=0)
            assert split.num_examples == len(split.examples) == n

    def test_dataclass_post_init_rejects_inconsistent_num_examples(self) -> None:
        with pytest.raises(ValueError):
            DatasetSplit(examples=[], num_examples=5, node_range=(5, 10), base_seed=0)


class TestWeightDensityPassThrough:
    @pytest.mark.parametrize("seed", range(5))
    def test_custom_weight_bounds_respected(self, seed: int) -> None:
        split = generate_dataset_split(
            num_examples=10, node_range=(5, 10), base_seed=seed,
            min_weight=3, max_weight=7,
        )
        for graph, _ in split.examples:
            for _, _, w in graph.edges:
                assert 3 <= w <= 7

    def test_default_weight_bounds_respected(self) -> None:
        split = generate_dataset_split(num_examples=10, node_range=(5, 10), base_seed=0)
        for graph, _ in split.examples:
            for _, _, w in graph.edges:
                assert 1 <= w <= 10

    def test_edge_density_changes_edge_count(self) -> None:
        split_sparse = generate_dataset_split(
            num_examples=20, node_range=(10, 10), base_seed=0, edge_density=0.2
        )
        split_dense = generate_dataset_split(
            num_examples=20, node_range=(10, 10), base_seed=0, edge_density=0.8
        )
        avg_sparse = sum(len(g.edges) for g, _ in split_sparse.examples) / 20
        avg_dense = sum(len(g.edges) for g, _ in split_dense.examples) / 20
        assert avg_dense > avg_sparse

    def test_spanning_tree_mode_when_no_density_given(self) -> None:
        split = generate_dataset_split(num_examples=10, node_range=(8, 8), base_seed=0)
        for graph, _ in split.examples:
            assert len(graph.edges) == graph.num_nodes - 1

    def test_num_edges_mode_respected(self) -> None:
        split = generate_dataset_split(
            num_examples=10, node_range=(8, 8), base_seed=0, num_edges=12
        )
        for graph, _ in split.examples:
            assert len(graph.edges) == 12

    def test_edge_density_is_safe_with_wide_node_range(self) -> None:
        split = generate_dataset_split(
            num_examples=20, node_range=(5, 20), base_seed=0, edge_density=0.4
        )
        assert split.num_examples == 20


class TestSeedScheme:
    _MULTIPLIER = 1_000_000

    def test_per_example_seeds_are_distinct(self) -> None:
        seeds = [0 * self._MULTIPLIER + i for i in range(50)]
        assert len(seeds) == len(set(seeds))

    def test_seeds_differ_across_base_seeds(self) -> None:
        n = 100
        seeds_0 = {0 * self._MULTIPLIER + i for i in range(n)}
        seeds_1 = {1 * self._MULTIPLIER + i for i in range(n)}
        assert seeds_0.isdisjoint(seeds_1)

    def test_graph_seed_field_matches_derived_seed(self) -> None:
        base_seed = 5
        split = generate_dataset_split(
            num_examples=10, node_range=(5, 10), base_seed=base_seed
        )
        for i, (graph, _) in enumerate(split.examples):
            assert graph.seed == base_seed * self._MULTIPLIER + i
"""
Phase 2: Correctness Verification Tests
Tests for query block reordering bugfix
"""
import pytest
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.graph_generator import Graph, generate_random_connected_graph
from data.tokenizer import GraphTokenizer


@pytest.fixture
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


class TestPermutationProperty:
    """Phase 2.2: Verify permutation property (markers at front, edge block unchanged)"""

    def test_permutation_property(self, tokenizer: GraphTokenizer) -> None:
        """Verify markers are at front, edge block follows unchanged."""
        graphs = [
            Graph(
                num_nodes=5,
                edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
                source=0,
                target=3,
                seed=42,
            ),
            Graph(
                num_nodes=10,
                edges=[(i, i + 1, 1) for i in range(9)],
                source=0,
                target=9,
                seed=99,
            ),
        ]

        for graph in graphs:
            encoded = tokenizer.encode_graph(graph)

            # Verify query block at [0, 1, 2, 3]
            assert (
                encoded[0] == tokenizer.src_token_id
            ), f"Query block not at front: src_token at position {encoded.index(tokenizer.src_token_id)}"
            assert (
                encoded[2] == tokenizer.dst_token_id
            ), f"Query block not at front: dst_token at position {encoded.index(tokenizer.dst_token_id)}"

            # Verify edge block starts at position 4
            edge_block_start = 4
            edge_token = encoded[edge_block_start]
            is_valid_edge_token = (
                edge_token in [tokenizer.token_to_id[f"node_{i}"] for i in range(50)]
                or edge_token in [tokenizer.token_to_id[f"weight_{w}"] for w in range(1, 11)]
            )
            assert (
                is_valid_edge_token
            ), f"Edge block doesn't start at position 4, got token {edge_token}"

            # Verify total length unchanged
            expected_length = 3 * len(graph.edges) + 4
            assert (
                len(encoded) == expected_length
            ), f"Length formula violated: {len(encoded)} != {expected_length}"


class TestMultisetPreservation:
    """Phase 2.3: Multiset preservation check (Counter invariant)"""

    def test_multiset_preserved(self, tokenizer: GraphTokenizer) -> None:
        """Verify token multiset is identical (permutation proof)."""
        graphs = [
            Graph(
                num_nodes=5,
                edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
                source=0,
                target=3,
                seed=42,
            ),
            Graph(
                num_nodes=6,
                edges=[(0, 1, 5), (1, 2, 3), (2, 3, 1), (3, 4, 2)],
                source=0,
                target=4,
                seed=55,
            ),
        ]

        for graph in graphs:
            encoded = tokenizer.encode_graph(graph)

            # Manually reconstruct old layout (edges + query block at end)
            old_layout = []
            for (u, v, w) in tokenizer._canonical_edges(graph):
                old_layout.append(tokenizer._node_token_id(u))
                old_layout.append(tokenizer._node_token_id(v))
                old_layout.append(tokenizer._weight_token_id(w))
            old_layout.append(tokenizer.src_token_id)
            old_layout.append(tokenizer._node_token_id(graph.source))
            old_layout.append(tokenizer.dst_token_id)
            old_layout.append(tokenizer._node_token_id(graph.target))

            # Verify multiset equality
            assert Counter(encoded) == Counter(old_layout), (
                f"Multiset mismatch:\n"
                f"  New layout: {Counter(encoded)}\n"
                f"  Old layout: {Counter(old_layout)}"
            )

    def test_multiset_edge_tokens_preserved_in_order(
        self, tokenizer: GraphTokenizer
    ) -> None:
        """Verify edge tokens are in canonical order (not just multiset)."""
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 3), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        encoded = tokenizer.encode_graph(graph)

        # Extract edge block (positions 4 onward)
        edge_block = encoded[4:]

        # Manually reconstruct canonical edges
        canonical_edges = tokenizer._canonical_edges(graph)
        expected_edge_tokens = []
        for (u, v, w) in canonical_edges:
            expected_edge_tokens.append(tokenizer._node_token_id(u))
            expected_edge_tokens.append(tokenizer._node_token_id(v))
            expected_edge_tokens.append(tokenizer._weight_token_id(w))

        assert (
            edge_block == expected_edge_tokens
        ), f"Edge tokens not in canonical order:\n  Got: {edge_block}\n  Expected: {expected_edge_tokens}"


class TestLengthInvariant:
    """Phase 2.4: Length invariant verification (3E+4 formula)"""

    def test_length_invariant_small_graphs(self, tokenizer: GraphTokenizer) -> None:
        """Verify length formula 3E + 4 holds for small graphs."""
        # Create graphs with specific edge counts
        test_cases = [
            (2, 1, "Two nodes, one edge"),
            (3, 3, "Three nodes, three edges"),
            (5, 10, "Five nodes, ten edges"),
            (8, 15, "Eight nodes, fifteen edges"),
        ]

        for num_nodes, target_edges, description in test_cases:
            graph = generate_random_connected_graph(
                num_nodes=num_nodes,
                seed=42,
                num_edges=target_edges,
                min_weight=1,
                max_weight=10,
            )

            encoded = tokenizer.encode_graph(graph)
            num_edges = len(graph.edges)
            expected_length = 3 * num_edges + 4

            assert (
                len(encoded) == expected_length
            ), f"{description}: E={num_edges}, len={len(encoded)}, expected={expected_length}"

    def test_length_invariant_random_graphs(self, tokenizer: GraphTokenizer) -> None:
        """Verify length formula holds for randomly generated graphs."""
        for seed in range(10):
            graph = generate_random_connected_graph(
                num_nodes=10,
                seed=seed,
                edge_density=0.5,
                min_weight=1,
                max_weight=10,
            )
            encoded = tokenizer.encode_graph(graph)

            num_edges = len(graph.edges)
            expected_length = 3 * num_edges + 4

            assert (
                len(encoded) == expected_length
            ), f"Seed {seed}: E={num_edges}, len={len(encoded)}, expected={expected_length}"


class TestQueryBlockPosition:
    """Additional verification: Query block at consistent positions"""

    def test_query_block_always_at_positions_0_through_3(
        self, tokenizer: GraphTokenizer
    ) -> None:
        """Query block must always be at positions 0-3 regardless of E."""
        for seed in range(5):
            graph = generate_random_connected_graph(
                num_nodes=15,
                seed=seed,
                edge_density=0.4,
                min_weight=1,
                max_weight=10,
            )
            encoded = tokenizer.encode_graph(graph)

            # Query block at [0, 1, 2, 3]
            assert encoded[0] == tokenizer.src_token_id, f"Seed {seed}: SRC not at position 0"
            assert (
                encoded[1] == tokenizer._node_token_id(graph.source)
            ), f"Seed {seed}: src_node not at position 1"
            assert encoded[2] == tokenizer.dst_token_id, f"Seed {seed}: DST not at position 2"
            assert (
                encoded[3] == tokenizer._node_token_id(graph.target)
            ), f"Seed {seed}: dst_node not at position 3"

    def test_edge_block_starts_at_position_4(self, tokenizer: GraphTokenizer) -> None:
        """Edge block must always start at position 4."""
        for seed in range(5):
            graph = generate_random_connected_graph(
                num_nodes=12,
                seed=seed,
                edge_density=0.5,
                min_weight=1,
                max_weight=10,
            )
            encoded = tokenizer.encode_graph(graph)

            # Edge block starts at position 4
            edge_block_start = encoded[4]
            canonical_edges = tokenizer._canonical_edges(graph)

            # First edge should be (u, v, w) → node_u, node_v, weight_w
            first_u, first_v, first_w = canonical_edges[0]
            expected_first_token = tokenizer._node_token_id(first_u)

            assert (
                edge_block_start == expected_first_token
            ), f"Seed {seed}: Edge block doesn't start at position 4"

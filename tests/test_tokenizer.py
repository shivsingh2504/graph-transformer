import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.graph_generator import Graph
from data.dijkstra import ShortestPath
from data.tokenizer import GraphTokenizer

@pytest.fixture
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


@pytest.fixture
def simple_graph() -> Graph:
    return Graph(
        num_nodes=4,
        edges=[
            (0, 1, 3),
            (0, 2, 2),
            (0, 3, 5),
            (2, 3, 1),
        ],
        source=0,
        target=3,
        seed=42,
    )


@pytest.fixture
def simple_path() -> ShortestPath:
    return ShortestPath(path=[0, 2, 3], cost=3)


class TestVocabDeterminism:
    def test_default_vocab_size(self, tokenizer: GraphTokenizer) -> None:
        assert tokenizer.vocab_size == 65

    def test_custom_weight_range_vocab_size(self) -> None:
        t = GraphTokenizer(min_weight=1, max_weight=5)
        assert t.vocab_size == 60

    def test_special_token_ids_are_fixed(self, tokenizer: GraphTokenizer) -> None:
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.src_token_id == 3
        assert tokenizer.dst_token_id == 4

    def test_node_token_ids_start_at_5(self, tokenizer: GraphTokenizer) -> None:
        assert tokenizer.token_to_id["node_0"] == 5
        assert tokenizer.token_to_id["node_49"] == 54

    def test_weight_token_ids_start_after_nodes(self, tokenizer: GraphTokenizer) -> None:
        assert tokenizer.token_to_id["weight_1"] == 55
        assert tokenizer.token_to_id["weight_10"] == 64

    def test_two_instantiations_have_identical_mapping(self) -> None:
        t1 = GraphTokenizer(min_weight=1, max_weight=10)
        t2 = GraphTokenizer(min_weight=1, max_weight=10)
        assert t1.token_to_id == t2.token_to_id
        assert t1.id_to_token == t2.id_to_token
        assert t1.vocab_size == t2.vocab_size

    def test_id_to_token_is_inverse_of_token_to_id(self, tokenizer: GraphTokenizer) -> None:
        for tok, idx in tokenizer.token_to_id.items():
            assert tokenizer.id_to_token[idx] == tok

    def test_no_duplicate_ids(self, tokenizer: GraphTokenizer) -> None:
        ids = list(tokenizer.token_to_id.values())
        assert len(ids) == len(set(ids))


class TestEncodeGraph:
    def test_output_length_matches_formula(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        num_unique_edges = len(simple_graph.edges)
        expected_len = 3 * num_unique_edges + 4
        assert len(encoded) == expected_len

    def test_src_dst_markers_present_in_output(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        assert tokenizer.src_token_id in encoded
        assert tokenizer.dst_token_id in encoded

    def test_src_and_target_nodes_follow_markers(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        # Query block at fixed positions [0, 1, 2, 3]
        assert encoded[0] == tokenizer.src_token_id
        assert encoded[1] == tokenizer.token_to_id[f"node_{simple_graph.source}"]
        assert encoded[2] == tokenizer.dst_token_id
        assert encoded[3] == tokenizer.token_to_id[f"node_{simple_graph.target}"]

    def test_src_dst_markers_at_front(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        assert encoded[0] == tokenizer.src_token_id
        assert encoded[2] == tokenizer.dst_token_id

    def test_query_block_at_front(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        """Query block (SRC, src_node, DST, dst_node) is at positions 0-3."""
        encoded = tokenizer.encode_graph(simple_graph)
        # Positions 0-3: query block
        assert encoded[0] == tokenizer.src_token_id
        assert encoded[1] == tokenizer._node_token_id(simple_graph.source)
        assert encoded[2] == tokenizer.dst_token_id
        assert encoded[3] == tokenizer._node_token_id(simple_graph.target)
        
        # Edge block follows
        assert len(encoded) == 3 * len(simple_graph.edges) + 4

    def test_bos_eos_not_in_encode_graph(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        assert tokenizer.bos_token_id not in encoded
        assert tokenizer.eos_token_id not in encoded

    def test_no_cost_in_encode_graph(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        for tid in encoded:
            assert 0 <= tid < tokenizer.vocab_size, (
                f"Token id {tid} is out of vocab range [0, {tokenizer.vocab_size})"
            )

    def test_deterministic_output(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        assert tokenizer.encode_graph(simple_graph) == tokenizer.encode_graph(simple_graph)

    def test_empty_edges_raises(self, tokenizer: GraphTokenizer) -> None:
        bad_graph = Graph(num_nodes=1, edges=[], source=0, target=0, seed=0)
        with pytest.raises(ValueError, match="no edges"):
            tokenizer.encode_graph(bad_graph)


class TestEncodePath:
    def test_starts_with_bos(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        assert encoded[0] == tokenizer.bos_token_id

    def test_ends_with_eos(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        assert encoded[-1] == tokenizer.eos_token_id

    def test_length_is_path_plus_two(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        assert len(encoded) == len(simple_path.path) + 2

    def test_no_cost_in_encode_path(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        for tid in encoded:
            assert 0 <= tid < tokenizer.vocab_size

    def test_no_weight_tokens_in_encode_path(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        weight_ids = {
            tokenizer.token_to_id[f"weight_{w}"]
            for w in range(tokenizer.min_weight, tokenizer.max_weight + 1)
        }
        encoded = tokenizer.encode_path(simple_path)
        for tid in encoded:
            assert tid not in weight_ids, (
                f"Weight token id {tid} found in encode_path output"
            )

    def test_src_dst_not_in_encode_path(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        assert tokenizer.src_token_id not in encoded
        assert tokenizer.dst_token_id not in encoded

    def test_middle_tokens_are_node_ids(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        for tid in encoded[1:-1]:
            tok = tokenizer.id_to_token[tid]
            assert tok.startswith("node_"), f"Expected node token, got '{tok}'"


class TestDecodePath:
    def test_roundtrip(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        assert tokenizer.decode_path(tokenizer.encode_path(simple_path)) == simple_path.path

    def test_strips_bos_and_eos(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        decoded = tokenizer.decode_path(tokenizer.encode_path(simple_path))
        assert decoded == simple_path.path
        assert len(decoded) == len(simple_path.path)

    def test_stops_at_first_eos(self, tokenizer: GraphTokenizer) -> None:
        path = ShortestPath(path=[0, 1, 2], cost=5)
        encoded = tokenizer.encode_path(path)
        padded = encoded + [tokenizer.pad_token_id] * 5
        decoded = tokenizer.decode_path(padded)
        assert decoded == [0, 1, 2]

    def test_no_bos_raises(self, tokenizer: GraphTokenizer) -> None:
        with pytest.raises(ValueError, match="BOS"):
            tokenizer.decode_path([tokenizer.eos_token_id, tokenizer.token_to_id["node_0"]])

    def test_empty_sequence_raises(self, tokenizer: GraphTokenizer) -> None:
        with pytest.raises(ValueError, match="BOS"):
            tokenizer.decode_path([])


class TestOODNodeId:
    def test_node_49_encodes_to_correct_id(self, tokenizer: GraphTokenizer) -> None:
        ood_path = ShortestPath(path=[0, 49], cost=1)
        encoded = tokenizer.encode_path(ood_path)
        assert tokenizer.token_to_id["node_49"] in encoded

    def test_node_49_roundtrip(self, tokenizer: GraphTokenizer) -> None:
        ood_path = ShortestPath(path=[0, 49], cost=1)
        assert tokenizer.decode_path(tokenizer.encode_path(ood_path)) == [0, 49]

    def test_node_49_in_graph(self, tokenizer: GraphTokenizer) -> None:
        ood_graph = Graph(
            num_nodes=50,
            edges=[(0, 49, 3)],
            source=0,
            target=49,
            seed=0,
        )
        encoded = tokenizer.encode_graph(ood_graph)
        assert tokenizer.token_to_id["node_49"] in encoded
        assert tokenizer.token_to_id["node_0"] in encoded

    def test_node_50_raises(self, tokenizer: GraphTokenizer) -> None:
        bad_path = ShortestPath(path=[0, 50], cost=1)
        with pytest.raises(ValueError, match="out of supported range"):
            tokenizer.encode_path(bad_path)


class TestNoTokenSpaceCollision:
    def test_weight_ids_dont_overlap_node_ids(self, tokenizer: GraphTokenizer) -> None:
        node_ids = {tokenizer.token_to_id[f"node_{n}"] for n in range(50)}
        weight_ids = {
            tokenizer.token_to_id[f"weight_{w}"]
            for w in range(1, 11)
        }
        assert node_ids.isdisjoint(weight_ids), (
            "Node token ids and weight token ids must not overlap"
        )

    def test_weight_ids_dont_overlap_special_ids(self, tokenizer: GraphTokenizer) -> None:
        special_ids = {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.src_token_id,
            tokenizer.dst_token_id,
        }
        weight_ids = {
            tokenizer.token_to_id[f"weight_{w}"]
            for w in range(1, 11)
        }
        assert special_ids.isdisjoint(weight_ids)

    def test_node_ids_dont_overlap_special_ids(self, tokenizer: GraphTokenizer) -> None:
        special_ids = {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.src_token_id,
            tokenizer.dst_token_id,
        }
        node_ids = {tokenizer.token_to_id[f"node_{n}"] for n in range(50)}
        assert special_ids.isdisjoint(node_ids)


class TestNoCostInOutput:
    def test_cost_value_not_a_valid_interpretation_in_graph(
        self, tokenizer: GraphTokenizer, simple_graph: Graph
    ) -> None:
        encoded = tokenizer.encode_graph(simple_graph)
        for tid in encoded:
            tok = tokenizer.id_to_token.get(tid, "")
            is_special = tok in {"[PAD]", "[BOS]", "[EOS]", "[SRC]", "[DST]"}
            is_node = tok.startswith("node_")
            is_weight = tok.startswith("weight_")
            assert is_special or is_node or is_weight, (
                f"Unexpected token '{tok}' (id={tid}) in encode_graph output"
            )

    def test_cost_value_not_a_valid_interpretation_in_path(
        self, tokenizer: GraphTokenizer, simple_path: ShortestPath
    ) -> None:
        encoded = tokenizer.encode_path(simple_path)
        for tid in encoded:
            tok = tokenizer.id_to_token.get(tid, "")
            is_bos_eos = tok in {"[BOS]", "[EOS]"}
            is_node = tok.startswith("node_")
            assert is_bos_eos or is_node, (
                f"Unexpected token '{tok}' (id={tid}) in encode_path output"
            )


class TestWeightBoundaryAccess:
    def test_min_max_weight_accessible(self, tokenizer: GraphTokenizer) -> None:
        assert tokenizer.min_weight == 1
        assert tokenizer.max_weight == 10
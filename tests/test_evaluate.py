from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from data.dataset_generator import DatasetSplit, generate_dataset_split
from data.dijkstra import ShortestPath, run_dijkstra
from data.graph_generator import Graph, generate_random_connected_graph
from data.tokenizer import GraphTokenizer
from model.model import Transformer

from eval.evaluate import (
    EvalResult,
    ExampleResult,
    _MAX_DECODE_LEN,
    _build_edge_set,
    _check_path,
    _decode_token_sequence,
    _greedy_decode,
    evaluate_example,
    evaluate_split,
)

DEVICE = torch.device("cpu")

@pytest.fixture(scope="module")
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


@pytest.fixture(scope="module")
def tiny_graph(tokenizer: GraphTokenizer) -> Graph:
    return generate_random_connected_graph(
        num_nodes=6, seed=7, edge_density=0.5,
        min_weight=1, max_weight=10,
    )


@pytest.fixture(scope="module")
def tiny_sp(tiny_graph: Graph) -> ShortestPath:
    return run_dijkstra(tiny_graph)


@pytest.fixture(scope="module")
def tiny_model(tokenizer: GraphTokenizer) -> Transformer:
    return Transformer(
        vocab_size=tokenizer.vocab_size,
        n_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0,
    ).to(DEVICE)


@pytest.fixture(scope="module")
def tiny_split() -> DatasetSplit:
    return generate_dataset_split(
        num_examples=10, node_range=(4, 6), base_seed=2, edge_density=0.5
    )



class TestAPIAssumptions:
    def test_transformer_has_encode_method(self, tiny_model: Transformer) -> None:
        assert hasattr(tiny_model, "encode")
        assert callable(tiny_model.encode)

    def test_transformer_has_decode_method(self, tiny_model: Transformer) -> None:
        assert hasattr(tiny_model, "decode")
        assert callable(tiny_model.decode)

    def test_transformer_encode_returns_3d_tensor(
        self, tiny_model: Transformer, tiny_graph: Graph, tokenizer: GraphTokenizer
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)]).to(DEVICE)
        with torch.no_grad():
            enc = tiny_model.encode(src_ids)
        assert isinstance(enc, torch.Tensor)
        assert enc.ndim == 3

    def test_transformer_decode_returns_logit_tensor(
        self, tiny_model: Transformer, tiny_graph: Graph, tokenizer: GraphTokenizer
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)]).to(DEVICE)
        tgt_ids = torch.tensor([[tokenizer.bos_token_id]]).to(DEVICE)
        with torch.no_grad():
            enc = tiny_model.encode(src_ids)
            logits = tiny_model.decode(tgt_ids, enc)
        assert isinstance(logits, torch.Tensor)
        assert logits.ndim == 3
        assert logits.shape[-1] == tokenizer.vocab_size

    def test_graph_has_adjacency_method(self, tiny_graph: Graph) -> None:
        assert hasattr(tiny_graph, "adjacency")
        assert callable(tiny_graph.adjacency)

    def test_graph_adjacency_returns_correct_structure(self, tiny_graph: Graph) -> None:
        adj = tiny_graph.adjacency()
        assert isinstance(adj, dict)
        for node_id in range(tiny_graph.num_nodes):
            assert node_id in adj
        for neighbours in adj.values():
            for item in neighbours:
                assert len(item) == 2
                neighbour, weight = item
                assert isinstance(neighbour, int)
                assert isinstance(weight, int)
                assert weight >= 1

    def test_adjacency_is_symmetric(self, tiny_graph: Graph) -> None:
        adj = tiny_graph.adjacency()
        for u, v, w in tiny_graph.edges:
            u_in_v = any(nbr == u and wt == w for nbr, wt in adj[v])
            assert u_in_v, (
                f"Edge ({u},{v},{w}): {u} with weight {w} not found in adj[{v}]"
            )
            v_in_u = any(nbr == v and wt == w for nbr, wt in adj[u])
            assert v_in_u, (
                f"Edge ({u},{v},{w}): {v} with weight {w} not found in adj[{u}]"
            )

    def test_max_decode_len_default_value(self) -> None:
        assert _MAX_DECODE_LEN == 60


class TestDecodeTokenSequence:
    def test_valid_sequence_returns_nodes(self, tokenizer: GraphTokenizer) -> None:
        ids = [
            tokenizer.bos_token_id,
            tokenizer.token_to_id["node_5"],
            tokenizer.token_to_id["node_3"],
            tokenizer.eos_token_id,
        ]
        assert _decode_token_sequence(ids, tokenizer) == [5, 3]

    def test_missing_bos_returns_none(self, tokenizer: GraphTokenizer) -> None:
        ids = [tokenizer.token_to_id["node_5"], tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_missing_eos_returns_none(self, tokenizer: GraphTokenizer) -> None:
        ids = [tokenizer.bos_token_id, tokenizer.token_to_id["node_5"]]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_empty_sequence_returns_none(self, tokenizer: GraphTokenizer) -> None:
        assert _decode_token_sequence([], tokenizer) is None

    def test_bos_immediately_followed_by_eos_returns_empty_list(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, tokenizer.eos_token_id]
        result = _decode_token_sequence(ids, tokenizer)
        assert result == []

    def test_weight_token_in_path_region_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [
            tokenizer.bos_token_id,
            tokenizer.token_to_id["weight_3"],
            tokenizer.eos_token_id,
        ]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_pad_token_in_path_region_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_src_token_in_path_region_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, tokenizer.src_token_id, tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_dst_token_in_path_region_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, tokenizer.dst_token_id, tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_completely_unknown_token_id_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, 99999, tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_sequence_of_multiple_nodes_correct_order(
        self, tokenizer: GraphTokenizer
    ) -> None:
        node_ids = [0, 3, 7, 12, 1]
        ids = (
            [tokenizer.bos_token_id]
            + [tokenizer.token_to_id[f"node_{n}"] for n in node_ids]
            + [tokenizer.eos_token_id]
        )
        assert _decode_token_sequence(ids, tokenizer) == node_ids

    def test_tokens_after_eos_are_ignored(self, tokenizer: GraphTokenizer) -> None:
        ids = [
            tokenizer.bos_token_id,
            tokenizer.token_to_id["node_2"],
            tokenizer.eos_token_id,
            tokenizer.token_to_id["node_9"],
        ]
        assert _decode_token_sequence(ids, tokenizer) == [2]

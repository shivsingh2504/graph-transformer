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
class TestBuildEdgeSet:
    def test_undirected_edges_present(self) -> None:
        graph = Graph(num_nodes=3, edges=[(0, 1, 2), (1, 2, 5)],
                      source=0, target=2, seed=0)
        es = _build_edge_set(graph)
        assert (0, 1) in es
        assert (1, 2) in es

    def test_canonical_min_max_ordering(self) -> None:
        graph = Graph(num_nodes=3, edges=[(2, 0, 3)], source=0, target=2, seed=0)
        es = _build_edge_set(graph)
        assert (0, 2) in es
        assert (2, 0) not in es

    def test_absent_edges_not_included(self) -> None:
        graph = Graph(num_nodes=3, edges=[(0, 1, 1)], source=0, target=1, seed=0)
        es = _build_edge_set(graph)
        assert (0, 2) not in es
        assert (1, 2) not in es

class TestCheckPath:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.graph = Graph(
            num_nodes=3,
            edges=[(0, 1, 1), (1, 2, 2), (0, 2, 5)],
            source=0, target=2, seed=0,
        )
        self.true_cost = 3

    def test_correct_optimal_path(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0, 1, 2], self.graph, self.true_cost)
        assert vp is True
        assert ce is True
        assert ev is True
        assert oc is True
        assert dc == 3

    def test_correct_but_suboptimal_path(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0, 2], self.graph, self.true_cost)
        assert vp is True
        assert ce is True
        assert ev is True
        assert oc is False
        assert dc == 5

    def test_cost_computed_from_edge_weights_not_from_any_token(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0, 1, 2], self.graph, true_cost=999)
        assert dc == 3
        assert oc is False

    def test_valid_path_true_when_edges_valid_but_wrong_source(self) -> None:
        vp, ce, ev, oc, dc = _check_path([1, 2], self.graph, self.true_cost)
        assert ev is True
        assert vp is True
        assert ce is False

    def test_valid_path_true_when_edges_valid_but_wrong_target(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0, 1], self.graph, self.true_cost)
        assert ev is True
        assert vp is True
        assert ce is False

    def test_invalid_edge_fails_edges_valid_and_valid_path(self) -> None:
        graph4 = Graph(
            num_nodes=4,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 3)],
            source=0, target=3, seed=0,
        )
        vp, ce, ev, oc, dc = _check_path([0, 1, 3], graph4, true_cost=6)
        assert ev is False
        assert vp is False
        assert dc is None

    def test_repeated_nodes_fail_valid_path_but_not_edges_valid(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0, 1, 0, 1, 2], self.graph, self.true_cost)
        assert ev is True
        assert vp is False
        assert ce is True
        assert dc == 5
        assert oc is False

    def test_empty_path(self) -> None:
        vp, ce, ev, oc, dc = _check_path([], self.graph, self.true_cost)
        assert vp is False
        assert ce is False
        assert ev is False
        assert oc is False
        assert dc is None

    def test_single_node_non_trivial_source_not_equal_target(self) -> None:
        vp, ce, ev, oc, dc = _check_path([0], self.graph, self.true_cost)
        assert vp is False
        assert ev is False

    def test_single_node_trivial_source_equals_target(self) -> None:
        graph = Graph(
            num_nodes=3,
            edges=[(0, 1, 1), (1, 2, 2)],
            source=1, target=1, seed=0,
        )
        vp, ce, ev, oc, dc = _check_path([1], graph, true_cost=0)
        assert vp is True
        assert ce is True
        assert ev is True
        assert oc is True
        assert dc == 0


class TestEvaluateExample:
    def test_returns_example_result_instance(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        assert isinstance(result, ExampleResult)

    def test_model_api_is_exercised_not_silently_bypassed(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        assert isinstance(result, ExampleResult)
        assert result.true_cost == tiny_sp.cost
        assert isinstance(result.decoded_nodes, list)
        for n in result.decoded_nodes:
            assert isinstance(n, int) and 0 <= n <= 49

    def test_true_cost_matches_dijkstra(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        assert result.true_cost == tiny_sp.cost

    def test_valid_and_optimal_implies_all_sub_checks(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        if result.valid_and_optimal:
            assert result.valid_path
            assert result.correct_endpoints
            assert result.edges_valid
            assert result.optimal_cost

    def test_valid_path_implies_edges_valid(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        if result.valid_path:
            assert result.edges_valid

    def test_decoded_cost_is_none_or_nonnegative(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_example(
            tiny_model, tiny_graph, tiny_sp, tokenizer, DEVICE, max_decode_len=30
        )
        if result.decoded_cost is not None:
            assert result.decoded_cost >= 0


class TestEvaluateSplit:
    def test_returns_eval_result_instance(
        self, tiny_model: Transformer,
        tiny_split: DatasetSplit,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_split(
            tiny_model, tiny_split, tokenizer, device=DEVICE, max_decode_len=20
        )
        assert isinstance(result, EvalResult)

    def test_num_examples_matches_split(
        self, tiny_model: Transformer,
        tiny_split: DatasetSplit,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_split(
            tiny_model, tiny_split, tokenizer, device=DEVICE, max_decode_len=20
        )
        assert result.num_examples == tiny_split.num_examples

    def test_per_example_length_matches(
        self, tiny_model: Transformer,
        tiny_split: DatasetSplit,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_split(
            tiny_model, tiny_split, tokenizer, device=DEVICE, max_decode_len=20
        )
        assert len(result.per_example) == tiny_split.num_examples

    def test_all_fractions_in_zero_one_range(
        self, tiny_model: Transformer,
        tiny_split: DatasetSplit,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_split(
            tiny_model, tiny_split, tokenizer, device=DEVICE, max_decode_len=20
        )
        assert 0.0 <= result.valid_and_optimal_fraction <= 1.0
        assert 0.0 <= result.edges_valid_fraction <= 1.0
        assert 0.0 <= result.valid_path_fraction <= 1.0
        assert 0.0 <= result.correct_endpoints_fraction <= 1.0
        assert 0.0 <= result.optimal_cost_fraction <= 1.0

    def test_summary_has_all_required_keys(
        self, tiny_model: Transformer,
        tiny_split: DatasetSplit,
        tokenizer: GraphTokenizer,
    ) -> None:
        result = evaluate_split(
            tiny_model, tiny_split, tokenizer, device=DEVICE, max_decode_len=20
        )
        s = result.summary()
        for key in (
            "valid_and_optimal_fraction",
            "edges_valid_fraction",
            "valid_path_fraction",
            "correct_endpoints_fraction",
            "optimal_cost_fraction",
            "num_examples",
            "num_valid_and_optimal",
        ):
            assert key in s, f"Missing summary key: {key}"

    def test_empty_split_returns_zero_fraction(
        self, tiny_model: Transformer,
        tokenizer: GraphTokenizer,
    ) -> None:
        empty = DatasetSplit(
            examples=[], num_examples=0, node_range=(4, 6), base_seed=0
        )
        result = evaluate_split(
            tiny_model, empty, tokenizer, device=DEVICE, max_decode_len=20
        )
        assert result.valid_and_optimal_fraction == 0.0
        assert result.num_examples == 0


class TestEvalResultProperties:
    def _make_result(self, flags: list) -> EvalResult:
        r = EvalResult()
        for f in flags:
            r.per_example.append(ExampleResult(
                decoded_nodes=[0, 1] if f else [],
                edges_valid=f,
                valid_path=f,
                correct_endpoints=f,
                optimal_cost=f,
                valid_and_optimal=f,
                decoded_cost=1 if f else None,
                true_cost=1,
            ))
        return r

    def test_all_valid(self) -> None:
        r = self._make_result([True, True, True, True])
        assert r.valid_and_optimal_fraction == pytest.approx(1.0)
        assert r.num_valid_and_optimal == 4

    def test_none_valid(self) -> None:
        r = self._make_result([False, False, False])
        assert r.valid_and_optimal_fraction == pytest.approx(0.0)
        assert r.num_valid_and_optimal == 0

    def test_mixed(self) -> None:
        r = self._make_result([True, False, True, False, True])
        assert r.valid_and_optimal_fraction == pytest.approx(3 / 5)
        assert r.num_valid_and_optimal == 3

    def test_empty_result(self) -> None:
        r = EvalResult()
        assert r.valid_and_optimal_fraction == 0.0
        assert r.num_examples == 0
        assert r.num_valid_and_optimal == 0

    def test_summary_values_consistent(self) -> None:
        r = self._make_result([True, True, False])
        s = r.summary()
        assert s["num_examples"] == pytest.approx(3.0)
        assert s["num_valid_and_optimal"] == pytest.approx(2.0)
        assert s["valid_and_optimal_fraction"] == pytest.approx(2 / 3)



class TestMalformedOutputRobustness:
    def test_no_eos_sequence_is_invalid(self, tokenizer: GraphTokenizer) -> None:
        ids = [tokenizer.bos_token_id] + [tokenizer.token_to_id["node_0"]] * 30
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_repeated_nodes_edges_valid_but_not_valid_path(self) -> None:
        graph = Graph(
            num_nodes=3,
            edges=[(0, 1, 1), (1, 2, 2)],
            source=0, target=2, seed=0,
        )
        vp, ce, ev, oc, dc = _check_path([0, 1, 0, 1, 2], graph, true_cost=3)
        assert ev is True
        assert vp is False
        assert dc == 5

    def test_out_of_range_token_id_returns_none(
        self, tokenizer: GraphTokenizer
    ) -> None:
        ids = [tokenizer.bos_token_id, 65535, tokenizer.eos_token_id]
        assert _decode_token_sequence(ids, tokenizer) is None

    def test_evaluate_example_graceful_on_all_pad_output(
        self,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        class AllPadModel(torch.nn.Module):
            def encode(self, src_ids: torch.Tensor, src_mask=None) -> torch.Tensor:
                B, S = src_ids.shape
                return torch.zeros(B, S, 32)

            def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor,
                       self_attn_mask=None, cross_attn_mask=None) -> torch.Tensor:
                B, T = tgt_ids.shape
                logits = torch.full((B, T, tokenizer.vocab_size), -1e9)
                logits[:, :, tokenizer.pad_token_id] = 1e9
                return logits

        result = evaluate_example(
            AllPadModel(), tiny_graph, tiny_sp, tokenizer, DEVICE,
            max_decode_len=10,
        )
        assert result.valid_and_optimal is False
        assert result.valid_path is False
        assert result.edges_valid is False

    def test_evaluate_example_graceful_on_never_eos(
        self,
        tiny_graph: Graph, tiny_sp: ShortestPath,
        tokenizer: GraphTokenizer,
    ) -> None:
        class NeverEosModel(torch.nn.Module):
            def encode(self, src_ids: torch.Tensor, src_mask=None) -> torch.Tensor:
                B, S = src_ids.shape
                return torch.zeros(B, S, 32)

            def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor,
                       self_attn_mask=None, cross_attn_mask=None) -> torch.Tensor:
                B, T = tgt_ids.shape
                logits = torch.full((B, T, tokenizer.vocab_size), -1e9)
                logits[:, :, tokenizer.token_to_id["node_0"]] = 1e9
                return logits

        result = evaluate_example(
            NeverEosModel(), tiny_graph, tiny_sp, tokenizer, DEVICE,
            max_decode_len=5,
        )
        assert result.valid_and_optimal is False
        assert result.valid_path is False



class TestGreedyDecode:
    def test_output_starts_with_bos(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tokenizer: GraphTokenizer,
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)])
        raw = _greedy_decode(tiny_model, src_ids, tokenizer, DEVICE, max_decode_len=20)
        assert len(raw) >= 1
        assert raw[0] == tokenizer.bos_token_id

    def test_output_length_bounded_by_max_decode_len(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tokenizer: GraphTokenizer,
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)])
        max_len = 5
        raw = _greedy_decode(tiny_model, src_ids, tokenizer, DEVICE, max_decode_len=max_len)
        assert len(raw) <= max_len + 1

    def test_stops_immediately_at_eos(self, tokenizer: GraphTokenizer) -> None:
        class ImmediateEosModel(torch.nn.Module):
            def encode(self, src_ids: torch.Tensor, src_mask=None) -> torch.Tensor:
                B, S = src_ids.shape
                return torch.zeros(B, S, 32)

            def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor,
                       self_attn_mask=None, cross_attn_mask=None) -> torch.Tensor:
                B, T = tgt_ids.shape
                logits = torch.full((B, T, tokenizer.vocab_size), -1e9)
                logits[:, :, tokenizer.eos_token_id] = 1e9
                return logits

        graph = generate_random_connected_graph(num_nodes=4, seed=0)
        src_ids = torch.tensor([tokenizer.encode_graph(graph)])
        raw = _greedy_decode(
            ImmediateEosModel(), src_ids, tokenizer, DEVICE, max_decode_len=10
        )
        assert raw == [tokenizer.bos_token_id, tokenizer.eos_token_id]

    def test_returns_list_of_ints(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tokenizer: GraphTokenizer,
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)])
        raw = _greedy_decode(tiny_model, src_ids, tokenizer, DEVICE, max_decode_len=10)
        assert isinstance(raw, list)
        for tok in raw:
            assert isinstance(tok, int)

    def test_all_token_ids_in_vocab_range(
        self, tiny_model: Transformer,
        tiny_graph: Graph, tokenizer: GraphTokenizer,
    ) -> None:
        src_ids = torch.tensor([tokenizer.encode_graph(tiny_graph)])
        raw = _greedy_decode(tiny_model, src_ids, tokenizer, DEVICE, max_decode_len=20)
        for tok in raw:
            assert 0 <= tok < tokenizer.vocab_size, (
                f"Token id {tok} is outside vocab range [0, {tokenizer.vocab_size})"
            )
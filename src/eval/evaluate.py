from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from data.dataset_generator import DatasetSplit
from data.dijkstra import ShortestPath
from data.graph_generator import Graph
from data.tokenizer import GraphTokenizer
from model.model import Transformer

_MAX_DECODE_LEN: int = 60


@dataclass(frozen=True)
class ExampleResult:
    decoded_nodes: List[int]
    edges_valid: bool
    valid_path: bool
    correct_endpoints: bool
    optimal_cost: bool
    valid_and_optimal: bool
    decoded_cost: Optional[int]
    true_cost: int


@dataclass
class EvalResult:
    per_example: List[ExampleResult] = field(default_factory=list)

    @property
    def num_examples(self) -> int:
        return len(self.per_example)

    @property
    def num_valid_and_optimal(self) -> int:
        return sum(1 for r in self.per_example if r.valid_and_optimal)

    @property
    def valid_and_optimal_fraction(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return self.num_valid_and_optimal / self.num_examples

    @property
    def edges_valid_fraction(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return sum(1 for r in self.per_example if r.edges_valid) / self.num_examples

    @property
    def valid_path_fraction(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return sum(1 for r in self.per_example if r.valid_path) / self.num_examples

    @property
    def correct_endpoints_fraction(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return sum(1 for r in self.per_example if r.correct_endpoints) / self.num_examples

    @property
    def optimal_cost_fraction(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return sum(1 for r in self.per_example if r.optimal_cost) / self.num_examples

    def summary(self) -> Dict[str, float]:
        return {
            "valid_and_optimal_fraction": self.valid_and_optimal_fraction,
            "edges_valid_fraction": self.edges_valid_fraction,
            "valid_path_fraction": self.valid_path_fraction,
            "correct_endpoints_fraction": self.correct_endpoints_fraction,
            "optimal_cost_fraction": self.optimal_cost_fraction,
            "num_examples": float(self.num_examples),
            "num_valid_and_optimal": float(self.num_valid_and_optimal),
        }


def _greedy_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    tokenizer: GraphTokenizer,
    device: torch.device,
    max_decode_len: int,
) -> List[int]:
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.to(device)
        encoder_output = model.encode(src_ids)

        generated: List[int] = [tokenizer.bos_token_id]

        for _ in range(max_decode_len):
            tgt_tensor = torch.tensor(
                [generated], dtype=torch.long, device=device
            )
            logits = model.decode(tgt_tensor, encoder_output)
            next_token_id: int = int(logits[0, -1].argmax().item())
            generated.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

    return generated


def _decode_token_sequence(
    token_ids: List[int],
    tokenizer: GraphTokenizer,
) -> Optional[List[int]]:
    if not token_ids:
        return None
    if token_ids[0] != tokenizer.bos_token_id:
        return None

    nodes: List[int] = []
    for tid in token_ids[1:]:
        if tid == tokenizer.eos_token_id:
            return nodes
        tok = tokenizer.id_to_token.get(tid)
        if tok is None:
            return None
        if not tok.startswith("node_"):
            return None
        try:
            node_id = int(tok[len("node_"):])
        except ValueError:
            return None
        nodes.append(node_id)

    return None


def _build_edge_set(graph: Graph) -> Set[Tuple[int, int]]:
    return {(min(u, v), max(u, v)) for u, v, _ in graph.edges}


def _check_path(
    nodes: List[int],
    graph: Graph,
    true_cost: int,
) -> Tuple[bool, bool, bool, bool, Optional[int]]:
    if len(nodes) == 0:
        return False, False, False, False, None

    source: int = graph.source
    target: int = graph.target

    correct_endpoints: bool = (nodes[0] == source and nodes[-1] == target)

    if len(nodes) == 1:
        trivial: bool = (source == target and nodes[0] == source)
        decoded_cost_single: Optional[int] = 0 if trivial else None
        optimal_cost_single: bool = (0 == true_cost) if trivial else False
        return trivial, correct_endpoints, trivial, optimal_cost_single, decoded_cost_single

    adj: Dict[int, List[Tuple[int, int]]] = graph.adjacency()
    edge_set: Set[Tuple[int, int]] = _build_edge_set(graph)

    edges_valid: bool = True
    decoded_cost: Optional[int] = 0

    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        canonical = (min(u, v), max(u, v))

        if canonical not in edge_set:
            edges_valid = False
            decoded_cost = None
            break

        for neighbour, weight in adj[u]:
            if neighbour == v:
                decoded_cost = decoded_cost + weight  # type: ignore[operator]
                break
        else:
            raise AssertionError(
                f"Edge {canonical} is present in edge_set but {v} not found "
                f"in adj[{u}].  graph.adjacency() appears asymmetric — "
                f"data pipeline error."
            )

    is_simple: bool = len(set(nodes)) == len(nodes)
    valid_path: bool = edges_valid and is_simple

    optimal_cost: bool = (decoded_cost == true_cost) if decoded_cost is not None else False

    return valid_path, correct_endpoints, edges_valid, optimal_cost, decoded_cost


def _invalid_result(true_cost: int) -> ExampleResult:
    return ExampleResult(
        decoded_nodes=[],
        edges_valid=False,
        valid_path=False,
        correct_endpoints=False,
        optimal_cost=False,
        valid_and_optimal=False,
        decoded_cost=None,
        true_cost=true_cost,
    )


def evaluate_example(
    model: Transformer,
    graph: Graph,
    shortest_path: ShortestPath,
    tokenizer: GraphTokenizer,
    device: torch.device,
    max_decode_len: int = _MAX_DECODE_LEN,
) -> ExampleResult:
    true_cost: int = shortest_path.cost

    try:
        src_ids_list: List[int] = tokenizer.encode_graph(graph)
    except ValueError:
        return _invalid_result(true_cost)

    src_ids = torch.tensor([src_ids_list], dtype=torch.long)

    raw_tokens: List[int] = _greedy_decode(
        model, src_ids, tokenizer, device, max_decode_len
    )

    nodes: Optional[List[int]] = _decode_token_sequence(raw_tokens, tokenizer)
    if nodes is None:
        return _invalid_result(true_cost)

    valid_path, correct_endpoints, edges_valid, optimal_cost, decoded_cost = (
        _check_path(nodes, graph, true_cost)
    )

    valid_and_optimal: bool = valid_path and correct_endpoints and optimal_cost

    return ExampleResult(
        decoded_nodes=nodes,
        edges_valid=edges_valid,
        valid_path=valid_path,
        correct_endpoints=correct_endpoints,
        optimal_cost=optimal_cost,
        valid_and_optimal=valid_and_optimal,
        decoded_cost=decoded_cost,
        true_cost=true_cost,
    )


def evaluate_split(
    model: Transformer,
    split: DatasetSplit,
    tokenizer: GraphTokenizer,
    *,
    device: Optional[torch.device] = None,
    max_decode_len: int = _MAX_DECODE_LEN,
) -> EvalResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    result = EvalResult()
    for graph, sp in split.examples:
        result.per_example.append(
            evaluate_example(model, graph, sp, tokenizer, device, max_decode_len)
        )

    return result
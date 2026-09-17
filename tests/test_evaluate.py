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

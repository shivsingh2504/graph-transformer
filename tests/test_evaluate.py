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

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from model.model import Transformer
from train.train import (
    GraphPathDataset,
    _make_masks,
    _train_epoch,
    _val_epoch,
    collate_fn,
    make_dataloader,
    train_model,
    _linear_warmup_schedule,
)

DEVICE = torch.device("cpu")
PAD_ID = 0   # confirmed: GraphTokenizer.pad_token_id == 0


@pytest.fixture(scope="module")
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


@pytest.fixture(scope="module")
def tiny_splits(tokenizer: GraphTokenizer):
    train_split = generate_dataset_split(
        num_examples=16, node_range=(5, 8), base_seed=0, edge_density=0.5
    )
    val_split = generate_dataset_split(
        num_examples=8, node_range=(5, 8), base_seed=1, edge_density=0.5
    )
    return train_split, val_split


def _tiny_model(vocab_size: int) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    ).to(DEVICE)

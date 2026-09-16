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

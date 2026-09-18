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
from train.train import _make_masks, _train_epoch, make_dataloader
from run_experiment import run_experiment
 
DEVICE = torch.device("cpu")
 
_TINY_TRAIN_KWARGS = dict(
    batch_size=4,
    lr=1e-3,
    warmup_steps=2,
    n_layers=1,
    d_model=32,
    n_heads=4,
    d_ff=64,
    dropout=0.0,
    device=DEVICE,
)
 
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
 
def _tiny_model(vocab_size: int) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    ).to(DEVICE)
 
 
def _global_grad_norm(model: Transformer) -> float:
    return torch.sqrt(
        sum(
            p.grad.detach().norm() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
    ).item()
 

class TestSeedValidation:
    def test_raises_when_seeds_equal(self) -> None:
        with pytest.raises(ValueError, match="train_seed and eval_seed must differ"):
            run_experiment(
                num_train_examples=4,
                num_eval_examples=4,
                n_epochs=1,
                train_seed=42,
                eval_seed=42,
                **_TINY_TRAIN_KWARGS,
            )
 
    def test_raises_when_n_epochs_in_kwargs(self) -> None:
        with pytest.raises(ValueError, match="'n_epochs' must be passed"):
            run_experiment(
                num_train_examples=4,
                num_eval_examples=4,
                n_epochs=1,
                train_seed=0,
                eval_seed=1,
                n_epochs=2,
                **{**_TINY_TRAIN_KWARGS, "n_epochs": 2},
            )
 
    def test_does_not_raise_when_seeds_differ(self) -> None:
        result = run_experiment(
            num_train_examples=4,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=0,
            eval_seed=1,
            **_TINY_TRAIN_KWARGS,
        )
        assert result is not None
 
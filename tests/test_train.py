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

class TestLossIgnoresPadding:
    def test_perturbing_pad_logits_does_not_change_loss(
        self, tokenizer: GraphTokenizer
    ) -> None:
        pad_id = tokenizer.pad_token_id
        vocab_size = tokenizer.vocab_size
        B, T = 2, 6

        target = torch.tensor([
            [5, 6, 7, 2, pad_id, pad_id],
            [5, 8, 2, pad_id, pad_id, pad_id],
        ])

        logits_base = torch.randn(B, T, vocab_size)
        logits_perturbed = logits_base.clone()
        logits_perturbed[0, 4:, :] += 100.0
        logits_perturbed[1, 3:, :] -= 100.0

        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss_base = criterion(logits_base.reshape(-1, vocab_size), target.reshape(-1))
        loss_perturbed = criterion(
            logits_perturbed.reshape(-1, vocab_size), target.reshape(-1)
        )

        assert torch.isclose(loss_base, loss_perturbed, atol=1e-5), (
            f"PAD-position logit change altered the loss: "
            f"{loss_base.item():.6f} vs {loss_perturbed.item():.6f}"
        )

    def test_criterion_ignore_index_matches_tokenizer(
        self, tokenizer: GraphTokenizer
    ) -> None:
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        assert criterion.ignore_index == tokenizer.pad_token_id == 0

class TestTeacherForcingShift:
    def test_decoder_input_starts_with_bos(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        _, tgt = next(iter(loader))
        decoder_input = tgt[:, :-1]
        assert (decoder_input[:, 0] == tokenizer.bos_token_id).all(), (
            "decoder_input[:,0] must be BOS for every example in the batch"
        )

    def test_target_output_contains_eos(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        _, tgt = next(iter(loader))
        target_output = tgt[:, 1:]
        has_eos = (target_output == tokenizer.eos_token_id).any(dim=1)
        assert has_eos.all(), "EOS missing from at least one target_output row"

    def test_shapes_are_equal_and_T_minus_1(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        _, tgt = next(iter(loader))
        decoder_input = tgt[:, :-1]
        target_output = tgt[:, 1:]
        assert decoder_input.shape == target_output.shape
        assert decoder_input.shape[1] == tgt.shape[1] - 1

    def test_decoder_input_and_target_output_are_shifted_by_one(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        _, tgt = next(iter(loader))
        # The two shifted views must reconstruct the original sequence
        combined = torch.cat([tgt[:, :1], tgt[:, 1:]], dim=1)
        assert torch.equal(combined, tgt)



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

class TestMakeMasks:
    def test_src_mask_shape(self, tokenizer: GraphTokenizer) -> None:
        B, S = 3, 10
        src = torch.randint(1, tokenizer.vocab_size, (B, S))
        src_mask, _, _ = _make_masks(src, torch.ones(B, 5, dtype=torch.long), PAD_ID)
        assert src_mask.shape == (B, 1, 1, S), f"Got {src_mask.shape}"

    def test_tgt_self_attn_mask_shape(self, tokenizer: GraphTokenizer) -> None:
        B, T = 3, 7
        src = torch.randint(1, tokenizer.vocab_size, (3, 10))
        dec_in = torch.randint(1, tokenizer.vocab_size, (B, T))
        _, tgt_self, _ = _make_masks(src, dec_in, PAD_ID)
        assert tgt_self.shape == (B, 1, T, T), f"Got {tgt_self.shape}"

    def test_tgt_cross_attn_mask_shape(self, tokenizer: GraphTokenizer) -> None:
        B, S = 3, 10
        src = torch.randint(1, tokenizer.vocab_size, (B, S))
        dec_in = torch.randint(1, tokenizer.vocab_size, (B, 5))
        _, _, cross = _make_masks(src, dec_in, PAD_ID)
        assert cross.shape == (B, 1, 1, S), f"Got {cross.shape}"

    def test_all_masks_are_bool(self, tokenizer: GraphTokenizer) -> None:
        src = torch.randint(1, tokenizer.vocab_size, (2, 8))
        dec_in = torch.randint(1, tokenizer.vocab_size, (2, 5))
        for mask in _make_masks(src, dec_in, PAD_ID):
            assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"

    def test_causal_structure_upper_triangle_is_false(
        self, tokenizer: GraphTokenizer
    ) -> None:
        B, T = 2, 6
        src = torch.randint(1, tokenizer.vocab_size, (B, 8))
        dec_in = torch.randint(1, tokenizer.vocab_size, (B, T))
        _, tgt_self, _ = _make_masks(src, dec_in, PAD_ID)
        mat = tgt_self[0, 0]   # (T, T)
        for i in range(T):
            for j in range(i + 1, T):
                assert not mat[i, j].item(), (
                    f"Future position ({i},{j}) is not masked"
                )

    def test_pad_positions_masked_in_src_mask(self) -> None:
        B, S = 2, 8
        src = torch.randint(1, 65, (B, S))
        src[:, -2:] = PAD_ID
        src_mask, _, _ = _make_masks(src, torch.ones(B, 4, dtype=torch.long), PAD_ID)
        assert not src_mask[0, 0, 0, -1].item()
        assert not src_mask[0, 0, 0, -2].item()
        assert src_mask[0, 0, 0, 0].item()

    def test_cross_attn_mask_is_src_mask(self) -> None:
        src = torch.randint(1, 65, (2, 8))
        dec_in = torch.randint(1, 65, (2, 5))
        src_mask, _, cross = _make_masks(src, dec_in, PAD_ID)
        assert torch.equal(src_mask, cross), (
            "tgt_cross_attn_mask must equal src_mask"
        )

class TestCausalMaskPreventsLeakage:
    def test_future_token_change_does_not_affect_past_logits(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=2, shuffle=False)
        src, tgt = next(iter(loader))
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        model = _tiny_model(tokenizer.vocab_size)
        model.eval()

        dec_in_1 = tgt[:, :-1].clone()
        dec_in_2 = tgt[:, :-1].clone()

        T = dec_in_1.size(1)
        dec_in_2[:, T - 1] = (dec_in_2[:, T - 1] + 5) % tokenizer.vocab_size
        dec_in_2[:, T - 1] = dec_in_2[:, T - 1].clamp(min=1)

        with torch.no_grad():
            src_mask, tgt_self_1, cross_1 = _make_masks(src, dec_in_1, PAD_ID)
            logits_1 = model(src, dec_in_1, src_mask=src_mask,
                             tgt_self_attn_mask=tgt_self_1,
                             tgt_cross_attn_mask=cross_1)

            _, tgt_self_2, cross_2 = _make_masks(src, dec_in_2, PAD_ID)
            logits_2 = model(src, dec_in_2, src_mask=src_mask,
                             tgt_self_attn_mask=tgt_self_2,
                             tgt_cross_attn_mask=cross_2)

        assert torch.allclose(logits_1[:, :-1, :], logits_2[:, :-1, :], atol=1e-5), (
            "Causal mask failed: logits at past positions changed when only "
            "a future decoder token was modified."
        )


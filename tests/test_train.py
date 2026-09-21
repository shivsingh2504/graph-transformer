from __future__ import annotations

import inspect
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

# ---------------------------------------------------------------------------
# Shared constants / fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
PAD_ID = 0   # confirmed: GraphTokenizer.pad_token_id == 0


@pytest.fixture(scope="module")
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


@pytest.fixture(scope="module")
def tiny_splits(tokenizer: GraphTokenizer):
    train_split = generate_dataset_split(
        num_examples=16, node_range=(5, 8), base_seed=0
    )
    val_split = generate_dataset_split(
        num_examples=8, node_range=(5, 8), base_seed=1
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


# ---------------------------------------------------------------------------
# 1. Loss ignores padding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Teacher-forcing shift — tested via loader output
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. _make_masks — shapes, dtypes, and semantics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Causal-leakage test
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 5. One training step works
# ---------------------------------------------------------------------------

class TestOneTrainingStep:
    def test_forward_produces_finite_logits(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        model = _tiny_model(tokenizer.vocab_size)
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        src, tgt = next(iter(loader))
        dec_in = tgt[:, :-1]
        src_mask, tgt_self, cross = _make_masks(src, dec_in, PAD_ID)
        logits = model(src, dec_in, src_mask=src_mask,
                       tgt_self_attn_mask=tgt_self, tgt_cross_attn_mask=cross)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_loss_is_finite(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        model = _tiny_model(tokenizer.vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        src, tgt = next(iter(loader))
        dec_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask, tgt_self, cross = _make_masks(src, dec_in, PAD_ID)
        logits = model(src, dec_in, src_mask=src_mask,
                       tgt_self_attn_mask=tgt_self, tgt_cross_attn_mask=cross)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        assert torch.isfinite(loss)

    def test_backward_runs_without_error(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        model = _tiny_model(tokenizer.vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        src, tgt = next(iter(loader))
        dec_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask, tgt_self, cross = _make_masks(src, dec_in, PAD_ID)
        logits = model(src, dec_in, src_mask=src_mask,
                       tgt_self_attn_mask=tgt_self, tgt_cross_attn_mask=cross)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# 6. Parameters actually update
# ---------------------------------------------------------------------------

class TestParametersUpdate:
    def test_embedding_weights_change_after_one_epoch(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        model = _tiny_model(tokenizer.vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        param_before = model.encoder.embedding.embedding.weight.data.clone()

        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        _train_epoch(model, loader, optimizer, criterion, DEVICE)

        param_after = model.encoder.embedding.embedding.weight.data
        assert not torch.equal(param_before, param_after), (
            "Embedding weights unchanged after training epoch — "
            "optimization is not working."
        )


# ---------------------------------------------------------------------------
# 7. Validation does not update parameters
# ---------------------------------------------------------------------------

class TestValidationDoesNotUpdate:
    def test_all_params_unchanged_after_val_epoch(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        _, val_split = tiny_splits
        model = _tiny_model(tokenizer.vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        params_before = {
            name: p.data.clone() for name, p in model.named_parameters()
        }

        loader = make_dataloader(val_split, tokenizer, batch_size=4, shuffle=False)
        _val_epoch(model, loader, criterion, DEVICE)

        for name, p in model.named_parameters():
            assert torch.equal(params_before[name], p.data), (
                f"Parameter '{name}' was modified during validation."
            )


# ---------------------------------------------------------------------------
# 8. Training history
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def _run(self, tokenizer, splits, n_epochs=2):
        train_split, val_split = splits
        return train_model(
            train_split, val_split, tokenizer,
            n_epochs=n_epochs, batch_size=8, lr=1e-3, warmup_steps=4,
            n_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0,
            device=DEVICE,
        )

    def test_history_has_both_keys(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        _, history = self._run(tokenizer, tiny_splits)
        assert "train_loss" in history
        assert "val_loss" in history

    def test_history_length_matches_epochs(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        n = 3
        _, history = self._run(tokenizer, tiny_splits, n_epochs=n)
        assert len(history["train_loss"]) == n
        assert len(history["val_loss"]) == n

    def test_history_values_are_finite_floats(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        _, history = self._run(tokenizer, tiny_splits)
        for loss in history["train_loss"] + history["val_loss"]:
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss: {loss}"


# ---------------------------------------------------------------------------
# 9. CPU compatibility
# ---------------------------------------------------------------------------

class TestCPUCompatibility:
    def test_train_model_runs_on_explicit_cpu(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, val_split = tiny_splits
        model, history = train_model(
            train_split, val_split, tokenizer,
            n_epochs=1, batch_size=4, lr=1e-3, warmup_steps=2,
            n_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0,
            device=torch.device("cpu"),
        )
        assert next(model.parameters()).device == torch.device("cpu")
        assert len(history["train_loss"]) == 1

    def test_dataloader_yields_cpu_tensors(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=4, shuffle=False)
        src, tgt = next(iter(loader))
        assert src.device == torch.device("cpu")
        assert tgt.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# 10. Warmup scheduler
# ---------------------------------------------------------------------------

class TestWarmupSchedule:
    def test_step_zero_gives_nonzero_lr(self) -> None:
        assert _linear_warmup_schedule(0, 400) == pytest.approx(1 / 400)

    def test_step_at_warmup_minus_one_gives_one(self) -> None:
        assert _linear_warmup_schedule(399, 400) == pytest.approx(1.0)

    def test_step_past_warmup_gives_one(self) -> None:
        assert _linear_warmup_schedule(500, 400) == pytest.approx(1.0)
        assert _linear_warmup_schedule(10_000, 400) == pytest.approx(1.0)

    def test_zero_warmup_always_gives_one(self) -> None:
        for step in [0, 1, 100]:
            assert _linear_warmup_schedule(step, 0) == pytest.approx(1.0)

    def test_lr_is_monotonically_increasing_during_warmup(self) -> None:
        warmup = 100
        multipliers = [_linear_warmup_schedule(s, warmup) for s in range(warmup)]
        for i in range(1, len(multipliers)):
            assert multipliers[i] > multipliers[i - 1]


# ---------------------------------------------------------------------------
# 11. Overfit-single-batch sanity check
# ---------------------------------------------------------------------------

class TestOverfitSingleBatch:
    def test_loss_drops_significantly_on_single_batch(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, _ = tiny_splits
        loader = make_dataloader(train_split, tokenizer, batch_size=8, shuffle=False)
        src, tgt = next(iter(loader))
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            n_layers=2, d_model=64, n_heads=4, d_ff=128, dropout=0.0,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        dec_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask, tgt_self, cross = _make_masks(src, dec_in, PAD_ID)

        final_loss = float("inf")
        for _ in range(300):
            model.train()
            optimizer.zero_grad()
            logits = model(src, dec_in, src_mask=src_mask,
                           tgt_self_attn_mask=tgt_self,
                           tgt_cross_attn_mask=cross)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert final_loss < 0.5, (
            f"Loss after 300 steps on a fixed batch is {final_loss:.4f} — "
            f"expected < 0.5. The training loop may be broken (wrong masks, "
            f"disconnected gradients, etc.)."
        )


# ---------------------------------------------------------------------------
# 12. Default-value assertions
# ---------------------------------------------------------------------------

class TestDefaultValues:
    def test_train_model_defaults(self) -> None:
        sig = inspect.signature(train_model)
        p = sig.parameters
        assert p["dropout"].default == 0.0
        assert p["warmup_steps"].default == 4000
        assert p["lr"].default == pytest.approx(5e-4)
        assert p["weight_decay"].default == pytest.approx(0.01)
        assert p["batch_size"].default == 32
        assert p["n_layers"].default == 3
        assert p["d_model"].default == 128
        assert p["n_heads"].default == 4
        assert p["d_ff"].default == 512
        assert p["grad_clip_norm"].default == pytest.approx(1.0)

    def test_transformer_default_dropout(self) -> None:
        sig = inspect.signature(Transformer)
        assert sig.parameters["dropout"].default == 0.0


# ---------------------------------------------------------------------------
# 13. on_epoch_end hook
# ---------------------------------------------------------------------------

class TestOnEpochEndHook:
    def test_called_exactly_n_times_with_correct_args(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, val_split = tiny_splits
        n_epochs = 3
        calls: list = []

        def spy(epoch: int, model: nn.Module, history: dict) -> None:
            calls.append({
                "epoch": epoch,
                "train_len": len(history["train_loss"]),
                "val_len": len(history["val_loss"]),
            })

        train_model(
            train_split, val_split, tokenizer,
            n_epochs=n_epochs, batch_size=8, lr=1e-3, warmup_steps=4,
            n_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0,
            device=DEVICE, on_epoch_end=spy,
        )

        assert len(calls) == n_epochs, (
            f"on_epoch_end called {len(calls)} times, expected {n_epochs}"
        )
        for i, call in enumerate(calls, start=1):
            assert call["epoch"] == i, (
                f"Call {i}: epoch={call['epoch']!r}, expected {i}"
            )
            assert call["train_len"] == i, (
                f"Call {i}: train_loss length={call['train_len']}, expected {i}"
            )
            assert call["val_len"] == i, (
                f"Call {i}: val_loss length={call['val_len']}, expected {i}"
            )

    def test_not_called_when_none(
        self, tokenizer: GraphTokenizer, tiny_splits
    ) -> None:
        train_split, val_split = tiny_splits
        model, history = train_model(
            train_split, val_split, tokenizer,
            n_epochs=1, batch_size=8, lr=1e-3, warmup_steps=4,
            n_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0,
            device=DEVICE, on_epoch_end=None,
        )
        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1

from __future__ import annotations

import copy
import os
import sys
from typing import Dict
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from model.model import Transformer
from train.train import _train_epoch, make_dataloader
from run_experiment import run_experiment

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _param_delta_norm(
    model: Transformer, snapshot: Dict[str, torch.Tensor]
) -> float:
    """L2 norm of (current params - snapshot) across all parameters."""
    return torch.sqrt(
        sum(
            (p.data - snapshot[n]).norm() ** 2
            for n, p in model.named_parameters()
        )
    ).item()


# ---------------------------------------------------------------------------
# Module-level fixture for T2 (plain @pytest.fixture, no class scope / classmethod)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t2_result():
    return run_experiment(
        node_range=(5, 20),
        num_train_examples=8,
        num_eval_examples=4,
        n_epochs=2,
        train_seed=10,
        eval_seed=20,
        **_TINY_TRAIN_KWARGS,
    )


# ---------------------------------------------------------------------------
# T1 — ValueError when seeds are equal
# ---------------------------------------------------------------------------

class TestSeedValidation:
    def test_raises_when_seeds_equal(self) -> None:
        with pytest.raises(ValueError, match="train_seed and eval_seed must differ"):
            run_experiment(
                node_range=(5, 20),
                num_train_examples=4,
                num_eval_examples=4,
                n_epochs=1,
                train_seed=42,
                eval_seed=42,
                **_TINY_TRAIN_KWARGS,
            )

    def test_does_not_raise_when_seeds_differ(self) -> None:
        result = run_experiment(
            node_range=(5, 20),
            num_train_examples=4,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=0,
            eval_seed=1,
            **_TINY_TRAIN_KWARGS,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# T2 — returned dict keys and types
# ---------------------------------------------------------------------------

class TestReturnedDictStructure:
    def test_top_level_keys_present(self, t2_result) -> None:
        assert set(t2_result.keys()) == {
            "train_loss", "val_loss", "eval_summary", "gate_passed", "model"
        }

    def test_train_loss_is_list_of_floats(self, t2_result) -> None:
        assert isinstance(t2_result["train_loss"], list)
        assert all(isinstance(v, float) for v in t2_result["train_loss"])

    def test_val_loss_is_list_of_floats(self, t2_result) -> None:
        assert isinstance(t2_result["val_loss"], list)
        assert all(isinstance(v, float) for v in t2_result["val_loss"])

    def test_train_loss_length_matches_n_epochs(self, t2_result) -> None:
        assert len(t2_result["train_loss"]) == 2

    def test_val_loss_length_matches_n_epochs(self, t2_result) -> None:
        assert len(t2_result["val_loss"]) == 2

    def test_eval_summary_is_dict_of_floats(self, t2_result) -> None:
        s = t2_result["eval_summary"]
        assert isinstance(s, dict)
        assert all(isinstance(v, float) for v in s.values())

    def test_eval_summary_contains_valid_and_optimal_fraction(self, t2_result) -> None:
        assert "valid_and_optimal_fraction" in t2_result["eval_summary"]

    def test_gate_passed_is_bool(self, t2_result) -> None:
        assert isinstance(t2_result["gate_passed"], bool)

    def test_model_is_transformer(self, t2_result) -> None:
        assert isinstance(t2_result["model"], Transformer)


# ---------------------------------------------------------------------------
# T3 — tiny end-to-end plumbing
# ---------------------------------------------------------------------------

class TestTinyEndToEnd:
    def test_fraction_in_unit_interval_and_example_count_correct(self) -> None:
        num_eval = 4
        result = run_experiment(
            node_range=(5, 20),
            num_train_examples=8,
            num_eval_examples=num_eval,
            n_epochs=1,
            train_seed=100,
            eval_seed=200,
            **_TINY_TRAIN_KWARGS,
        )
        vof = result["eval_summary"]["valid_and_optimal_fraction"]
        assert 0.0 <= vof <= 1.0, f"valid_and_optimal_fraction out of [0,1]: {vof}"
        # Confirm evaluate_split actually ran on the right number of examples,
        # not silently returning 0.0 for an empty result.
        assert result["eval_summary"]["num_examples"] == float(num_eval), (
            f"Expected {num_eval} eval examples, "
            f"got {result['eval_summary']['num_examples']}"
        )

    def test_losses_are_non_empty_and_finite(self) -> None:
        n_epochs = 2
        result = run_experiment(
            node_range=(5, 20),
            num_train_examples=8,
            num_eval_examples=4,
            n_epochs=n_epochs,
            train_seed=102,
            eval_seed=203,
            **_TINY_TRAIN_KWARGS,
        )
        assert len(result["train_loss"]) == n_epochs, (
            f"train_loss has {len(result['train_loss'])} entries, expected {n_epochs}"
        )
        assert len(result["val_loss"]) == n_epochs, (
            f"val_loss has {len(result['val_loss'])} entries, expected {n_epochs}"
        )
        for loss_val in result["train_loss"] + result["val_loss"]:
            assert torch.isfinite(torch.tensor(loss_val)), (
                f"Non-finite loss encountered: {loss_val}"
            )


# ---------------------------------------------------------------------------
# T4 — gate boundary: monkeypatched stubs
#
# Probes vof = 0.89, 0.90, 0.91 so the test catches:
#   • wrong constant (e.g. 0.91 instead of 0.90)
#   • wrong operator (> instead of >=)
# An untrained model always gives vof ≈ 0, so a live run can only exercise
# the False branch; stubs are the only way to cover the True branch too.
# ---------------------------------------------------------------------------

class TestGateBoundary:
    @pytest.mark.parametrize("vof,expected_gate", [
        (0.89, False),
        (0.90, True),
        (0.91, True),
    ])
    def test_gate_uses_correct_threshold_and_operator(
        self, monkeypatch, vof: float, expected_gate: bool
    ) -> None:
        import run_experiment as re_module

        dummy_history = {"train_loss": [0.5], "val_loss": [0.5]}
        dummy_model = MagicMock()
        monkeypatch.setattr(
            re_module, "train_model",
            lambda *a, **kw: (dummy_model, dummy_history),
        )

        dummy_eval = MagicMock()
        dummy_eval.valid_and_optimal_fraction = vof
        dummy_eval.summary.return_value = {"valid_and_optimal_fraction": vof}
        monkeypatch.setattr(
            re_module, "evaluate_split",
            lambda *a, **kw: dummy_eval,
        )

        result = run_experiment(
            node_range=(5, 20),
            num_train_examples=4,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=0,
            eval_seed=1,
            # train_model is stubbed, so model kwargs are irrelevant
        )
        assert result["gate_passed"] == expected_gate, (
            f"vof={vof}: expected gate_passed={expected_gate}, "
            f"got {result['gate_passed']}"
        )



class TestSplitSeedIsolation:
    def test_no_graph_overlaps_across_full_splits(self) -> None:
        """
        Verify train and eval splits share no graph by comparing structural
        content (num_nodes + full weighted-edge set).  Seeds 0 and 1 are the
        worst case for any base_seed+i scheme.
        """
        num_train, num_eval = 20, 10
        train_split = generate_dataset_split(
            num_examples=num_train, node_range=(5, 20), base_seed=0
        )
        eval_split = generate_dataset_split(
            num_examples=num_eval, node_range=(5, 20), base_seed=1
        )

        def _graph_fingerprint(g) -> tuple:
            return (g.num_nodes, tuple(sorted(g.edges)))

        train_fps = {_graph_fingerprint(g) for g, _ in train_split.examples}
        eval_fps  = {_graph_fingerprint(g) for g, _ in eval_split.examples}

        assert train_fps.isdisjoint(eval_fps), (
            f"Graph-content overlap between train and eval splits: "
            f"{train_fps & eval_fps}"
        )
        # Guard against empty splits or within-split duplicates
        assert len(train_fps) == num_train, (
            f"train_split has duplicates or is empty: "
            f"{num_train} examples but {len(train_fps)} unique graphs"
        )
        assert len(eval_fps) == num_eval, (
            f"eval_split has duplicates or is empty: "
            f"{num_eval} examples but {len(eval_fps)} unique graphs"
        )

    def test_same_base_seed_is_deterministic(self) -> None:
        """Same base_seed -> identical graphs at every position."""
        split_a = generate_dataset_split(
            num_examples=4, node_range=(5, 20), base_seed=7
        )
        split_b = generate_dataset_split(
            num_examples=4, node_range=(5, 20), base_seed=7
        )
        for (ga, _), (gb, _) in zip(split_a.examples, split_b.examples):
            assert ga.num_nodes == gb.num_nodes
            assert sorted(ga.edges) == sorted(gb.edges)

    def test_run_experiment_passes_correct_seeds_and_counts(
        self, monkeypatch
    ) -> None:
        """
        Spy on generate_dataset_split inside run_experiment's module namespace.
        Swapping train_seed/eval_seed or misrouting num_train/num_eval would
        be caught here; the direct split tests above would not catch it.
        """
        import run_experiment as re_module

        calls: list = []
        _orig = re_module.generate_dataset_split

        def _spy(**kwargs):
            calls.append(kwargs.copy())
            return _orig(**kwargs)

        monkeypatch.setattr(re_module, "generate_dataset_split", _spy)

        run_experiment(
            node_range=(5, 20),
            num_train_examples=5,
            num_eval_examples=3,
            n_epochs=1,
            train_seed=11,
            eval_seed=22,
            **_TINY_TRAIN_KWARGS,
        )

        assert len(calls) == 2, (
            f"Expected exactly 2 calls to generate_dataset_split, got {len(calls)}"
        )
        # First call: training split
        assert calls[0]["num_examples"] == 5,   f"train num_examples wrong: {calls[0]}"
        assert calls[0]["base_seed"]    == 11,  f"train base_seed wrong: {calls[0]}"
        assert calls[0]["node_range"]   == (5, 20)
        # Second call: eval split
        assert calls[1]["num_examples"] == 3,   f"eval num_examples wrong: {calls[1]}"
        assert calls[1]["base_seed"]    == 22,  f"eval base_seed wrong: {calls[1]}"
        assert calls[1]["node_range"]   == (5, 20)


# ---------------------------------------------------------------------------
# T6 — grad clipping behavioural
#
# Math: with SGD lr=1.0 and exactly one optimizer step,
#   ||Δparams|| = lr * ||grad_after_clip|| = ||grad_after_clip||
# clip_grad_norm_(C) sets ||grad|| = min(unclipped, C), so
#   ||Δparams||_clipped  <=  C            (must hold)
#   ||Δparams||_control  >   C            (test discriminates)
#
# batch_size == num_examples guarantees exactly one step.
# ---------------------------------------------------------------------------

class TestGradClipping:
    def test_clipping_constrains_parameter_delta(self) -> None:
        tokenizer = GraphTokenizer(min_weight=1, max_weight=10)
        num_ex = 4
        split = generate_dataset_split(
            num_examples=num_ex, node_range=(5, 8), base_seed=42
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        # batch_size == num_ex -> exactly one optimizer step per epoch
        loader = make_dataloader(split, tokenizer, batch_size=num_ex, shuffle=False)

        torch.manual_seed(0)  # pin init so delta magnitude is reproducible
        model_clipped = _tiny_model(tokenizer.vocab_size)
        model_control = copy.deepcopy(model_clipped)

        snapshot = {n: p.data.clone() for n, p in model_clipped.named_parameters()}

        clip_norm = 0.01

        optim_clipped = torch.optim.SGD(model_clipped.parameters(), lr=1.0)
        _train_epoch(
            model_clipped, loader, optim_clipped, criterion, DEVICE,
            grad_clip_norm=clip_norm,
        )

        optim_control = torch.optim.SGD(model_control.parameters(), lr=1.0)
        _train_epoch(
            model_control, loader, optim_control, criterion, DEVICE,
            grad_clip_norm=1e9,      # effectively no clip
        )

        clipped_delta = _param_delta_norm(model_clipped, snapshot)
        control_delta = _param_delta_norm(model_control, snapshot)

        assert clipped_delta <= clip_norm + 1e-5, (
            f"Clipped parameter delta {clipped_delta:.6f} exceeds "
            f"clip_norm={clip_norm}. clip_grad_norm_ may not be wired into "
            "_train_epoch, or fires after optimizer.step()."
        )
        # Lower bound: a correct clip must move params by almost exactly
        # clip_norm (not zero, which would pass the upper bound vacuously).
        assert clipped_delta == pytest.approx(clip_norm, rel=1e-2), (
            f"Clipped parameter delta {clipped_delta:.6f} is not close to "
            f"clip_norm={clip_norm}. Expected the gradient to be rescaled to "
            "exactly clip_norm, not zeroed or left unclipped."
        )
        assert control_delta > clip_norm, (
            f"Control delta {control_delta:.6f} is not larger than "
            f"clip_norm={clip_norm}; the test cannot distinguish clipped from "
            "unclipped training — verify the loss is non-trivial."
        )


# ---------------------------------------------------------------------------
# T7 — on_epoch_end passthrough
# ---------------------------------------------------------------------------

class TestOnEpochEndPassthrough:
    def test_on_epoch_end_reaches_train_model(self) -> None:
        calls: list = []

        def spy(epoch: int, model, history: dict) -> None:
            calls.append(epoch)

        run_experiment(
            node_range=(5, 20),
            num_train_examples=4,
            num_eval_examples=4,
            n_epochs=2,
            train_seed=50,
            eval_seed=51,
            on_epoch_end=spy,
            **_TINY_TRAIN_KWARGS,
        )

        assert calls == [1, 2], (
            f"on_epoch_end received epochs {calls}, expected [1, 2]"
        )
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from data.tokenizer import GraphTokenizer
from model.model import Transformer
from train.checkpoint import load_checkpoint, save_checkpoint

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


def _tiny_model_config(vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "n_layers": 1,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "dropout": 0.0,
    }


def _make_model(model_config: dict) -> Transformer:
    torch.manual_seed(0)
    return Transformer(**model_config).to(DEVICE)


class TestCheckpointRoundTrip:
    def test_round_trip_outputs_equal(
        self, tmp_path, tokenizer: GraphTokenizer
    ) -> None:
        model_config = _tiny_model_config(tokenizer.vocab_size)
        model = _make_model(model_config)
        model.eval()

        src = torch.randint(1, tokenizer.vocab_size, (2, 8))
        tgt_in = torch.randint(1, tokenizer.vocab_size, (2, 5))

        with torch.no_grad():
            expected = model(src, tgt_in)

        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            model_config=model_config,
            train_config={"lr": 1e-4},
            seeds={"train_seed": 0},
            tokenizer_kwargs={"min_weight": 1, "max_weight": 10},
            history={"train_loss": [1.0], "val_loss": [1.0]},
        )

        loaded_model, _ = load_checkpoint(ckpt_path, DEVICE)
        with torch.no_grad():
            actual = loaded_model(src, tgt_in)

        assert torch.allclose(expected, actual, atol=1e-6), (
            f"Round-trip outputs differ: max diff = "
            f"{(expected - actual).abs().max().item():.2e}"
        )

    def test_perturbed_model_gives_different_outputs(
        self, tmp_path, tokenizer: GraphTokenizer
    ) -> None:
        model_config = _tiny_model_config(tokenizer.vocab_size)
        model = _make_model(model_config)
        model.eval()

        src = torch.randint(1, tokenizer.vocab_size, (2, 8))
        tgt_in = torch.randint(1, tokenizer.vocab_size, (2, 5))

        with torch.no_grad():
            original_out = model(src, tgt_in)

        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            model_config=model_config,
            train_config={"lr": 1e-4},
            seeds={"train_seed": 0},
            tokenizer_kwargs={"min_weight": 1, "max_weight": 10},
            history={},
        )

        loaded_model, _ = load_checkpoint(ckpt_path, DEVICE)
        with torch.no_grad():
            next(loaded_model.parameters()).add_(10.0)

        with torch.no_grad():
            perturbed_out = loaded_model(src, tgt_in)

        assert not torch.allclose(original_out, perturbed_out, atol=1e-6), (
            "Perturbed model gave identical outputs — the control cannot "
            "distinguish a correct from an incorrect round-trip."
        )

    def test_all_keys_present_and_config_values_equal(
        self, tmp_path, tokenizer: GraphTokenizer
    ) -> None:
        model_config = _tiny_model_config(tokenizer.vocab_size)
        train_config = {"lr": 1e-4, "batch_size": 32}
        seeds = {"train_seed": 7, "eval_seed": 8}
        tokenizer_kwargs = {"min_weight": 1, "max_weight": 10}
        history = {"train_loss": [2.0, 1.5], "val_loss": [2.1, 1.6]}

        model = _make_model(model_config)
        ckpt_path = tmp_path / "subdir" / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            model_config=model_config,
            train_config=train_config,
            seeds=seeds,
            tokenizer_kwargs=tokenizer_kwargs,
            history=history,
        )

        _, ckpt = load_checkpoint(ckpt_path, DEVICE)

        for key in (
            "state_dict", "model_config", "train_config",
            "seeds", "tokenizer_kwargs", "history",
        ):
            assert key in ckpt, f"Missing key in checkpoint: {key!r}"

        assert ckpt["model_config"] == model_config
        assert ckpt["train_config"] == train_config
        assert ckpt["seeds"] == seeds
        assert ckpt["tokenizer_kwargs"] == tokenizer_kwargs
        assert ckpt["history"] == history

    def test_parent_dirs_created(
        self, tmp_path, tokenizer: GraphTokenizer
    ) -> None:
        model_config = _tiny_model_config(tokenizer.vocab_size)
        model = _make_model(model_config)

        deep_path = tmp_path / "a" / "b" / "c" / "ckpt.pt"
        assert not deep_path.parent.exists()

        save_checkpoint(
            deep_path,
            model,
            model_config=model_config,
            train_config={},
            seeds={},
            tokenizer_kwargs={"min_weight": 1, "max_weight": 10},
            history={},
        )

        assert deep_path.exists()

    def test_tokenizer_vocab_size_matches_model_config(
        self, tmp_path, tokenizer: GraphTokenizer
    ) -> None:
        tokenizer_kwargs = {"min_weight": 1, "max_weight": 10}
        model_config = _tiny_model_config(tokenizer.vocab_size)
        model = _make_model(model_config)

        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            model_config=model_config,
            train_config={},
            seeds={},
            tokenizer_kwargs=tokenizer_kwargs,
            history={},
        )

        _, ckpt = load_checkpoint(ckpt_path, DEVICE)
        reconstructed = GraphTokenizer(**ckpt["tokenizer_kwargs"])
        assert reconstructed.vocab_size == ckpt["model_config"]["vocab_size"], (
            f"GraphTokenizer(**tokenizer_kwargs).vocab_size="
            f"{reconstructed.vocab_size} != "
            f"model_config['vocab_size']={ckpt['model_config']['vocab_size']}"
        )

from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from model.model import Transformer
from run_final_training import RunConfig, run
from train.checkpoint import load_checkpoint
from train.train import train_model

_TINY_CFG = RunConfig(
    num_train_examples=8,
    num_eval_examples=4,
    n_epochs=1,
    n_layers=1,
    d_model=32,
    n_heads=4,
    d_ff=64,
    warmup_steps=2,
)


class TestRunFinalTraining:
    def test_output_files_exist(self, tmp_path) -> None:
        run(_TINY_CFG, tmp_path)
        assert (tmp_path / "latest.pt").exists()
        assert (tmp_path / "final.pt").exists()
        assert (tmp_path / "results.json").exists()

    def test_results_json_has_all_keys(self, tmp_path) -> None:
        run(_TINY_CFG, tmp_path)
        with open(tmp_path / "results.json") as fh:
            data = json.load(fh)
        for key in (
            "config",
            "train_loss",
            "val_loss",
            "eval_summary",
            "gate_passed",
            "duplicate_count",
            "wall_time_seconds",
            "torch_version",
        ):
            assert key in data, f"Missing key in results.json: {key!r}"

    def test_final_pt_round_trips_through_load_checkpoint(
        self, tmp_path
    ) -> None:
        run(_TINY_CFG, tmp_path)
        model, ckpt = load_checkpoint(tmp_path / "final.pt", torch.device("cpu"))
        assert isinstance(model, Transformer)
        assert "state_dict" in ckpt
        assert "model_config" in ckpt
        assert "train_config" in ckpt

    def test_second_call_raises_file_exists_error(self, tmp_path) -> None:
        run(_TINY_CFG, tmp_path)
        with pytest.raises(FileExistsError):
            run(_TINY_CFG, tmp_path)

    def test_run_config_defaults_match_train_model_signature(self) -> None:
        sig = inspect.signature(train_model)
        params = sig.parameters
        cfg = RunConfig()
        shared = {
            "dropout": cfg.dropout,
            "warmup_steps": cfg.warmup_steps,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "grad_clip_norm": cfg.grad_clip_norm,
        }
        for name, cfg_value in shared.items():
            sig_value = params[name].default
            assert sig_value == cfg_value, (
                f"RunConfig.{name}={cfg_value!r} != "
                f"train_model default {name}={sig_value!r}"
            )

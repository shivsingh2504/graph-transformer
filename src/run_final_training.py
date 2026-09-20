from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from run_experiment import run_experiment
from train.checkpoint import save_checkpoint

_NODE_RANGE = (5, 20)


@dataclass(frozen=True)
class RunConfig:
    num_train_examples: int = 50_000
    num_eval_examples: int = 1_000
    n_epochs: int = 30
    train_seed: int = 0
    eval_seed: int = 1
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4_000
    grad_clip_norm: float = 1.0
    n_layers: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0


def _fingerprint(graph: Any) -> tuple:
    return (graph.num_nodes, tuple(sorted(graph.edges)), graph.source, graph.target)


def run(cfg: RunConfig, out_dir: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    final_pt = out_dir / "final.pt"
    results_json = out_dir / "results.json"

    if final_pt.exists() or results_json.exists():
        raise FileExistsError(
            f"Output already exists in {out_dir}: "
            f"remove final.pt and results.json before rerunning."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.train_seed)

    # Sanity-check splits — not used for training
    _train_split = generate_dataset_split(
        num_examples=cfg.num_train_examples,
        node_range=_NODE_RANGE,
        base_seed=cfg.train_seed,
    )
    _eval_split = generate_dataset_split(
        num_examples=cfg.num_eval_examples,
        node_range=_NODE_RANGE,
        base_seed=cfg.eval_seed,
    )
    assert len(_train_split.examples) == cfg.num_train_examples
    assert len(_eval_split.examples) == cfg.num_eval_examples

    train_fps = {_fingerprint(g) for g, _ in _train_split.examples}
    eval_fps = {_fingerprint(g) for g, _ in _eval_split.examples}
    duplicate_count = len(train_fps & eval_fps)
    print(f"Cross-split duplicate graphs (fingerprint collisions): {duplicate_count}")

    tokenizer_kwargs: Dict[str, int] = {"min_weight": 1, "max_weight": 10}
    _tok = GraphTokenizer(**tokenizer_kwargs)
    model_config: Dict[str, Any] = {
        "vocab_size": _tok.vocab_size,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "dropout": cfg.dropout,
    }
    train_config: Dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": cfg.warmup_steps,
        "grad_clip_norm": cfg.grad_clip_norm,
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
    }
    seeds: Dict[str, int] = {
        "train_seed": cfg.train_seed,
        "eval_seed": cfg.eval_seed,
        "torch_seed": cfg.train_seed,
    }

    def _on_epoch_end(
        epoch: int, model: nn.Module, history: Dict[str, List[float]]
    ) -> None:
        save_checkpoint(
            out_dir / "latest.pt",
            model,
            model_config=model_config,
            train_config=train_config,
            seeds=seeds,
            tokenizer_kwargs=tokenizer_kwargs,
            history=history,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time()
    result = run_experiment(
        num_train_examples=cfg.num_train_examples,
        num_eval_examples=cfg.num_eval_examples,
        n_epochs=cfg.n_epochs,
        train_seed=cfg.train_seed,
        eval_seed=cfg.eval_seed,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        grad_clip_norm=cfg.grad_clip_norm,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        device=device,
        on_epoch_end=_on_epoch_end,
    )
    wall_time = time.time() - t_start

    save_checkpoint(
        final_pt,
        result["model"],
        model_config=model_config,
        train_config=train_config,
        seeds=seeds,
        tokenizer_kwargs=tokenizer_kwargs,
        history={"train_loss": result["train_loss"], "val_loss": result["val_loss"]},
    )

    results_data: Dict[str, Any] = {
        "config": dataclasses.asdict(cfg),
        "train_loss": result["train_loss"],
        "val_loss": result["val_loss"],
        "eval_summary": result["eval_summary"],
        "gate_passed": result["gate_passed"],
        "duplicate_count": duplicate_count,
        "wall_time_seconds": wall_time,
        "torch_version": torch.__version__,
    }

    with open(results_json, "w") as fh:
        json.dump(results_data, fh, indent=2)

    return results_data


def main() -> None:
    run(RunConfig(), Path("checkpoints"))


if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    main()

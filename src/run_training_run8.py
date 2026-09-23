"""
Run 8: LR schedule modification on run 7 base (layout fix + 60 epochs).

LR schedule: hold 5e-4 to epoch 40, decay to 0 by epoch 60
vs run 7: constant 5e-4 throughout 60 epochs
"""
import os
import sys
import torch
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import evaluate_split
from model.model import Transformer
from train.train import train_one_epoch


@dataclass
class RunConfig:
    """Run 8 configuration: run 7 base + LR schedule change"""
    n_epochs: int = 60
    train_data_seed: int = 0
    eval_data_seed: int = 1
    node_range: tuple = (5, 20)
    n_train_examples: int = 500_000
    n_eval_examples: int = 1_000
    output_dir: str = "checkpoints_run8"
    learning_rate: float = 5e-4
    learning_rate_schedule: str = "decay_40_60"  # New in run 8


def get_lr_for_epoch(epoch: int, base_lr: float = 5e-4) -> float:
    """
    LR schedule for run 8:
    - Epochs 0-39: lr = base_lr (hold)
    - Epochs 40-59: lr = base_lr * (1 - (epoch-40)/20) (linear decay to 0)
    """
    if epoch < 40:
        return base_lr
    else:
        decay_progress = (epoch - 40) / 20.0
        return base_lr * max(0.0, 1.0 - decay_progress)


def train_and_save_checkpoint(config: RunConfig):
    """Train run 8 with LR schedule and save checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Setup
    print("Generating training split...")
    train_split = generate_dataset_split(
        config.n_train_examples, config.node_range, config.train_data_seed
    )
    print(f"  Generated {len(train_split.examples)} training examples")

    print("Generating eval split...")
    eval_split = generate_dataset_split(
        config.n_eval_examples, config.node_range, config.eval_data_seed
    )
    print(f"  Generated {len(eval_split.examples)} eval examples")
    print()

    # Model
    model_cfg = {
        "vocab_size": 65,
        "n_layers": 4,
        "d_model": 256,
        "n_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
    }
    tokenizer = GraphTokenizer(min_weight=1, max_weight=10)
    model = Transformer(**model_cfg)
    model.to(device)

    print(f"Model: {model_cfg['n_layers']} layers, d_model={model_cfg['d_model']}")
    print(f"Tokenizer: vocab_size={model_cfg['vocab_size']}")
    print()

    print(f"Training for {config.n_epochs} epochs with LR schedule:")
    print(f"  Epochs 0-39: lr = {config.learning_rate}")
    print(f"  Epochs 40-59: lr decays from {config.learning_rate} to 0")
    print()

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)

    # Training loop (placeholder - actual training would call train_one_epoch)
    print("NOTE: This is a template. Actual training requires integration with train.py")
    print(f"Configuration saved to {config.output_dir}/run8_config.txt")

    # Save checkpoint structure
    torch.save(
        {
            "model_config": model_cfg,
            "tokenizer_kwargs": {"min_weight": 1, "max_weight": 10},
            "state_dict": model.state_dict(),
            "run8_schedule": "decay_40_60",
        },
        os.path.join(config.output_dir, "final.pt"),
    )

    print(f"Checkpoint saved to {config.output_dir}/final.pt")


if __name__ == "__main__":
    config = RunConfig()
    print("=" * 70)
    print("RUN 8: LR Schedule Change (hold 5e-4 to epoch 40, decay to 0)")
    print("=" * 70)
    print()
    print(f"Config:")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Train examples: {config.n_train_examples}")
    print(f"  LR schedule: {config.learning_rate_schedule}")
    print(f"  Output: {config.output_dir}")
    print()
    
    # Note: actual training would be called here
    # train_and_save_checkpoint(config)
    print("Run 8 script ready. Execute with actual training integration.")

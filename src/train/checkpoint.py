from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from model.model import Transformer


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    seeds: Dict[str, Any],
    tokenizer_kwargs: Dict[str, Any],
    history: Dict[str, list],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_config": model_config,
        "train_config": train_config,
        "seeds": seeds,
        "tokenizer_kwargs": tokenizer_kwargs,
        "history": history,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path | str,
    device: torch.device,
) -> Tuple[Transformer, Dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint

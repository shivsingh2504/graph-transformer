"""Run 8: LR schedule modification (hold 5e-4 to epoch 40, decay to 0 by epoch 60).

Run 8 adds a learning rate decay schedule on top of Run 7:
- Epochs 0-39: LR holds at 5e-4 (after warmup)
- Epochs 40-59: LR decays linearly from 5e-4 to 0

Changes from Run 7:
- Learning rate schedule: adds lr_decay_start_epoch=40
- Number of epochs: 60 (no change from run 7)
- All other config inherited from Run 7

Run with:  .venv\Scripts\python srcun_training_run8.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from run_final_training import RunConfig, run

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = dataclasses.replace(RunConfig(), n_epochs=60, lr_decay_start_epoch=40)
    changed = {
        f.name: (getattr(RunConfig(), f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(RunConfig(), f.name) != getattr(cfg, f.name)
    }
    assert changed == {"n_epochs": (30, 60), "lr_decay_start_epoch": (None, 40)}, changed
    print(f"Changes vs run 5 baseline: {changed}")
    run(cfg, _ROOT / "checkpoints_run8")


if __name__ == "__main__":
    main()

"""Run 9: Node range extension (5-20 -> 5-30) with Run 8's LR schedule.

Run 9 extends the training node range from (5, 20) to (5, 30):
- Node range: (5, 30) — the key change
- Epochs: 60 (inherited from Run 8)
- LR schedule: hold 5e-4 to epoch 40, decay to 0 by epoch 60 (inherited from Run 8)
- All other hyperparameters: same as Run 8

Changes from Run 8:
- Node range: (5, 20) -> (5, 30)
- All other config identical to Run 8

Run with:  python src/run_training_run9.py
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
    cfg = dataclasses.replace(
        RunConfig(),
        node_range=(5, 30),
        n_epochs=60,
        lr_decay_start_epoch=40
    )
    changed = {
        f.name: (getattr(RunConfig(), f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(RunConfig(), f.name) != getattr(cfg, f.name)
    }
    expected = {
        "node_range": ((5, 20), (5, 30)),
        "n_epochs": (30, 60),
        "lr_decay_start_epoch": (None, 40)
    }
    assert changed == expected, f"Expected {expected}, got {changed}"
    print(f"Run 9 changes vs baseline: {changed}")
    run(cfg, _ROOT / "checkpoints_run9")


if __name__ == "__main__":
    main()

"""Run 6: identical to run 5 except n_epochs (30 -> 60).

Run 5 was undertrained: train loss 0.1224 vs val loss 0.1285 (no overfit gap),
val loss still falling over the last 10 epochs (0.1481 -> 0.1285), and the best
val loss was the final epoch. Runs 4 and 5 hold the locked distribution and the
500k-example train split fixed and differ only in epoch count (10 -> 30 took
valid_and_optimal from 46.3% to 83.6%), so epoch count is the variable with
direct evidence behind it.

The LR schedule is a linear warmup to a constant (`_linear_warmup_schedule` in
train/train.py returns 1.0 past warmup_steps), so it does not reshape when
n_epochs changes. warmup_steps stays at 4000 and every other field of RunConfig
is inherited unchanged, which keeps this a single-variable run.

Run with:  .venv\\Scripts\\python src\\run_training_run6.py
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
_N_EPOCHS_RUN6 = 60


def main() -> None:
    cfg = dataclasses.replace(RunConfig(), n_epochs=_N_EPOCHS_RUN6)
    changed = {
        f.name: (getattr(RunConfig(), f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(RunConfig(), f.name) != getattr(cfg, f.name)
    }
    assert changed == {"n_epochs": (30, _N_EPOCHS_RUN6)}, changed
    print(f"Single variable vs run 5: {changed}")
    run(cfg, _ROOT / "checkpoints_run6")


if __name__ == "__main__":
    main()

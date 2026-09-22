"""Run 7: Implements query-block-fixed-positions bugfix.

Run 7 trains with the new tokenizer layout where source/destination markers are
placed at fixed positions [0, 1, 2, 3] instead of at the end of the sequence.

Changes from Run 6:
- Tokenizer: src/data/tokenizer.py encode_graph() places query block first
- All other config inherited from Run 6: 500k examples, 60 epochs, same hyperparameters

Expected improvement:
- ID: maintain >= 84.1% (within -2 pts of run 6's 86.1%)
- OOD: endpoints >= 90% at N=21-30 sizes

Run with:  .venv\Scripts\python src\run_training_run7.py
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
_N_EPOCHS_RUN7 = 60


def main() -> None:
    cfg = dataclasses.replace(RunConfig(), n_epochs=_N_EPOCHS_RUN7)
    changed = {
        f.name: (getattr(RunConfig(), f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(RunConfig(), f.name) != getattr(cfg, f.name)
    }
    assert changed == {"n_epochs": (30, _N_EPOCHS_RUN7)}, changed
    print(f"Single variable vs run 5 baseline: {changed}")
    print("Changes from run 6: None (layout only, in tokenizer.py)")
    run(cfg, _ROOT / "checkpoints_run7")


if __name__ == "__main__":
    main()

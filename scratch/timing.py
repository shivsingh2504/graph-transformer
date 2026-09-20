import sys
import os
sys.path.append(os.path.abspath('src'))
import time
from pathlib import Path
from run_final_training import run, RunConfig

print("Starting short training slice (1 epoch) for Run 3 timing...")
cfg = RunConfig(n_epochs=1)
# Create a dummy run directory
out_dir = Path("scratch/checkpoints_run3_timing")
if out_dir.exists():
    import shutil
    shutil.rmtree(out_dir)

start = time.time()
res = run(cfg, out_dir)
duration = time.time() - start

print(f"Time for 1 epoch: {duration:.2f} seconds")
print(f"Projected time for 30 epochs: {duration * 30 / 60:.2f} minutes")

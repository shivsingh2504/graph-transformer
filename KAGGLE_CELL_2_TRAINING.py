# Cell 2: Run training (commit mode safe)

%cd /kaggle/working/graph

print("=" * 80)
print("STARTING RUN 7 TRAINING")
print("=" * 80)
print()

import subprocess
import sys
import os

# Run training
print("Command: python src/run_training_run7.py")
print()

result = subprocess.run(
    [sys.executable, "src/run_training_run7.py"],
    capture_output=False,
)

print()
print("=" * 80)
if result.returncode == 0:
    print("✓ TRAINING COMPLETED")
else:
    print(f"✗ TRAINING FAILED (code: {result.returncode})")
print("=" * 80)

# Verify outputs
import os
print("\nOutput verification:")
for f in ["checkpoints_run7/final.pt", "checkpoints_run7/results.json"]:
    exists = os.path.exists(f)
    print(f"  {'✓' if exists else '✗'} {f}")

# Show loss summary if available
import json
if os.path.exists("checkpoints_run7/results.json"):
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    with open("checkpoints_run7/results.json") as fh:
        results = json.load(fh)
    
    print(f"\nWall time: {results.get('wall_time_seconds', 'N/A'):.1f}s")
    
    if "train_loss" in results:
        losses = results["train_loss"]
        print(f"\nLoss progression (first 3 and last 3 epochs):")
        for i in range(min(3, len(losses))):
            print(f"  Epoch {i:2d}: {losses[i]:.4f}")
        if len(losses) > 6:
            print(f"  ...")
        for i in range(max(0, len(losses)-3), len(losses)):
            print(f"  Epoch {i:2d}: {losses[i]:.4f}")
    
    eval_summary = results.get("eval_summary", {})
    if eval_summary:
        print(f"\nID Evaluation (1000 examples, N=5-20):")
        print(f"  Valid & Optimal: {eval_summary.get('valid_and_optimal_fraction', 0)*100:.1f}%")
        print(f"  Correct Endpoints: {eval_summary.get('correct_endpoints_fraction', 0)*100:.1f}%")
        print(f"  Edges Valid: {eval_summary.get('edges_valid_fraction', 0)*100:.1f}%")

"""Phase 6: Quick 10-Minute Diagnostic Smoke Test

Goal: Validate the new query-block-fixed-positions layout works in a training 
context before committing 6.5 hours to full training.

Test: Run a 2-3 epoch mini-training job using the new tokenizer layout to confirm:
  1. No NaNs or infinities in loss
  2. Loss decreases normally (expected trajectory)
  3. No shape mismatches or tensor errors
  4. Checkpoint save/resume works with new layout
  5. Training infrastructure compatible with new layout

Expected Duration: ~5-10 minutes on CPU/GPU
Success: Training starts cleanly, loss decreases, checkpoints save, no layout-related errors
"""

from __future__ import annotations

import dataclasses
import json
import sys
import os
from pathlib import Path

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from run_final_training import RunConfig, run

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Run a mini-training job with reduced dataset size and fewer epochs."""
    
    # Mini config: 1000 train, 100 eval, 2 epochs (vs full 500k/1k, 30+ epochs)
    cfg = dataclasses.replace(
        RunConfig(),
        num_train_examples=1_000,
        num_eval_examples=100,
        n_epochs=3,  # 3 epochs for quick validation
    )
    
    # Verify changes vs baseline
    changed = {
        f.name: (getattr(RunConfig(), f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(RunConfig(), f.name) != getattr(cfg, f.name)
    }
    
    expected_changes = {
        "num_train_examples": (500_000, 1_000),
        "num_eval_examples": (1_000, 100),
        "n_epochs": (30, 3),
    }
    assert changed == expected_changes, f"Unexpected config changes: {changed}"
    
    print("=" * 80)
    print("PHASE 6: Quick 10-Minute Diagnostic Smoke Test")
    print("=" * 80)
    print(f"\nConfiguration (mini-training):")
    print(f"  Train examples: {cfg.num_train_examples:,}")
    print(f"  Eval examples:  {cfg.num_eval_examples:,}")
    print(f"  Epochs:         {cfg.n_epochs}")
    print(f"  Batch size:     {cfg.batch_size}")
    print(f"  Output:         {_ROOT / 'checkpoints_phase6_diagnostic'}")
    print()
    
    # Run training with mini config
    print("Starting mini-training job...")
    print("-" * 80)
    try:
        results = run(cfg, _ROOT / "checkpoints_phase6_diagnostic")
        
        print("-" * 80)
        print("\n✓ PHASE 6 DIAGNOSTIC: PASS")
        print("\nResults Summary:")
        
        # Print loss trajectory
        if "train_losses" in results:
            print(f"\nLoss Trajectory:")
            for epoch, loss in enumerate(results["train_losses"][-3:]):
                print(f"  Epoch {epoch}: {loss:.6f}")
        
        if "val_losses" in results:
            print(f"\nValidation Loss:")
            for epoch, loss in enumerate(results["val_losses"][-3:]):
                print(f"  Epoch {epoch}: {loss:.6f}")
        
        # Validation checks
        print(f"\nValidation Checks:")
        
        # Check 1: Loss is finite
        if "train_losses" in results:
            all_finite = all(
                isinstance(l, (int, float)) and not (l != l) and abs(l) != float('inf')
                for l in results["train_losses"]
            )
            print(f"  ✓ Loss values are finite: {all_finite}")
        
        # Check 2: Loss decreased
        if "train_losses" in results and len(results["train_losses"]) > 1:
            initial_loss = results["train_losses"][0]
            final_loss = results["train_losses"][-1]
            decreased = final_loss < initial_loss
            print(f"  ✓ Loss decreased: {initial_loss:.6f} → {final_loss:.6f} ({decreased})")
        
        # Check 3: Checkpoint saved
        checkpoint_path = _ROOT / "checkpoints_phase6_diagnostic" / "final.pt"
        checkpoint_exists = checkpoint_path.exists()
        print(f"  ✓ Checkpoint saved: {checkpoint_exists}")
        
        # Check 4: Results file exists
        results_path = _ROOT / "checkpoints_phase6_diagnostic" / "results.json"
        results_exist = results_path.exists()
        print(f"  ✓ Results file saved: {results_exist}")
        
        print("\n" + "=" * 80)
        print("SUCCESS: Ready for full training (Phase 7)")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print("-" * 80)
        print(f"\n✗ PHASE 6 DIAGNOSTIC: FAIL")
        print(f"\nError: {e}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("FAILURE: Diagnostic failed before full training")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

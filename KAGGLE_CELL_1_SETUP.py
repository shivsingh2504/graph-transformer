# Cell 1: Clone repo, checkout validated fix, verify diffs

%cd /kaggle/working
!rm -rf graph
!git clone https://github.com/shivsingh2504/graph-transformer graph
%cd /kaggle/working/graph

print("=" * 80)
print("CHECKING OUT VALIDATED FIX COMMIT")
print("=" * 80)

# Checkout the exact commit where the fix was validated
!git checkout -q 1a7701d
!git rev-parse HEAD

print("\nCommit history (last 5):")
!git log --oneline -5

print("\n" + "=" * 80)
print("VERIFYING FIX INTEGRITY")
print("=" * 80)

print("\n[A] Core modules (model, train, run_experiment) unchanged vs run 6?")
print("    (Should print NOTHING if clean)")
!git diff --stat d154c74 HEAD -- src/model src/train/train.py src/run_experiment.py

print("\n[B] Query-block fix present in tokenizer.py?")
print("    (Should print src/data/tokenizer.py with changes)")
!git diff --stat 044ad98 HEAD -- src/data/tokenizer.py

print("\n[C] run_training_run7.py exists?")
import os
exists = os.path.exists("src/run_training_run7.py")
print(f"    {'✓ YES' if exists else '✗ NO'}")

print("\n" + "=" * 80)
print("GPU AND ENVIRONMENT CHECK")
print("=" * 80)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print(f"✗ WARNING: Training on CPU will be VERY SLOW")
    print(f"  Check notebook settings: Accelerator > GPU T4 x2")

print(f"\nPyTorch: {torch.__version__}")

print("\n" + "=" * 80)
print("QUERY-BLOCK SANITY CHECK")
print("=" * 80)

import sys
sys.path.insert(0, "/kaggle/working/graph/src")
from data.graph_generator import Graph
from data.tokenizer import GraphTokenizer

tok = GraphTokenizer()
test_graph = Graph(
    num_nodes=5,
    edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
    source=0,
    target=3,
    seed=42
)
encoded = tok.encode_graph(test_graph)

print(f"\nTest graph (N=5, E=3):")
print(f"  Encoded length: {len(encoded)} (expected: 3*3+4=13)")
print(f"  Position [0]: SRC token? {encoded[0] == tok.src_token_id} (should be True)")
print(f"  Position [2]: DST token? {encoded[2] == tok.dst_token_id} (should be True)")
print(f"  Position [4]: Edge block starts? {encoded[4] in [tok.token_to_id[f'node_{i}'] for i in range(50)]} (should be True)")

if len(encoded) == 13 and encoded[0] == tok.src_token_id and encoded[2] == tok.dst_token_id:
    print("\n✓ QUERY-BLOCK FIX VERIFIED")
else:
    print("\n✗ QUERY-BLOCK ISSUE DETECTED")
    raise RuntimeError("Query-block layout not as expected")

print("\n" + "=" * 80)
print("PRE-TRAINING CHECKS: PASS")
print("=" * 80)

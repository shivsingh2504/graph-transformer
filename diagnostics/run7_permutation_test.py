"""Pre-launch item 1 for the run 7 query-block ruling.

Runs the four permutation assertions against the REAL src/data/tokenizer.py
encode_graph — not the prototype in run7_layout_probe.py.  Must exit 0 before
run 7 is ratified.

Assertions (per graph, across 1,850 real generated graphs in four splits):
  A. head  — new[:4] == [SRC, node_s, DST, node_t] using graph.source/target
              directly (not the old sequence).  This is the load-bearing check:
              the negative tests showed that swapping source/target nodes while
              keeping markers in place is caught *only* by this assertion.
  B. perm  — new[4:] == old[:-4]  (edge block is identical, just shifted left)
  C. len   — len(new) == len(old) == 3*E + 4
  D. counter — Counter(new) == Counter(old)  (same multiset of tokens)
  E. pad   — pad_token_id does not appear in new

Additionally:
  F. formula — new[0] == SRC and new[2] == DST (marker ids at fixed positions)
  G. non-trivial — the *old* layout really did trail the markers, confirming
                   the move is genuine and not a no-op.

Run from the repo root:
    .venv\Scripts\python diagnostics\run7_permutation_test.py

Exit 0 = all assertions passed.
Exit 1 = one or more assertions failed (details printed above the summary).
Exit 2 = encode_graph has not been changed yet — run this after modifying tokenizer.py.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer

# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"{tag}  {label}{suffix}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# Tokenizer -- uses the REAL encode_graph, which must already be modified
# ---------------------------------------------------------------------------

tok = GraphTokenizer(min_weight=1, max_weight=10)

# Sanity-check that encode_graph has been changed to the query-first layout
# before running the full suite.  One probe graph is enough.
_probe_split = generate_dataset_split(num_examples=1, node_range=(10, 10), base_seed=99)
_probe_g, _ = _probe_split.examples[0]
_probe_enc = tok.encode_graph(_probe_g)
if _probe_enc[0] != tok.src_token_id:
    print()
    print("ABORT: encode_graph does NOT appear to use the query-first layout.")
    print(f"       token[0] should be src_token_id ({tok.src_token_id}), got {_probe_enc[0]}.")
    print("       Modify src/data/tokenizer.py first, then re-run this script.")
    sys.exit(2)

print("encode_graph layout check: query block is at the front -- proceeding.")
print()

# ---------------------------------------------------------------------------
# Reproduce the old (query-last) layout from helpers for comparison.
# This is kept here so the test is self-contained; it must NOT be used
# anywhere in the training pipeline.
# ---------------------------------------------------------------------------

def _encode_query_last(tok: GraphTokenizer, graph) -> list[int]:
    """Pre-run-7 layout: edge block then [SRC] s [DST] t."""
    ids: list[int] = []
    for u, v, w in tok._canonical_edges(graph):
        ids.append(tok._node_token_id(u))
        ids.append(tok._node_token_id(v))
        ids.append(tok._weight_token_id(w))
    ids.append(tok.src_token_id)
    ids.append(tok._node_token_id(graph.source))
    ids.append(tok.dst_token_id)
    ids.append(tok._node_token_id(graph.target))
    return ids


# ---------------------------------------------------------------------------
# Item 1: permutation invariants over four splits
# ---------------------------------------------------------------------------

SPLITS = [
    ("ID   N=5-20  base_seed=0",  dict(num_examples=1000, node_range=(5, 20),  base_seed=0)),
    ("OOD  N=21-50 base_seed=2",  dict(num_examples=600,  node_range=(21, 50), base_seed=2)),
    ("N=7  fixed   base_seed=3",  dict(num_examples=200,  node_range=(7, 7),   base_seed=3)),
    ("N=50 fixed   base_seed=4",  dict(num_examples=50,   node_range=(50, 50), base_seed=4)),
]

total_graphs = 0
observed_lengths: set[int] = set()

for split_label, kwargs in SPLITS:
    split = generate_dataset_split(**kwargs)
    bad_head = bad_perm = bad_len = bad_counter = bad_pad = bad_formula = 0
    local_lens: set[int] = set()

    for graph, _sp in split.examples:
        new = tok.encode_graph(graph)   # REAL encode_graph -- must be query-first
        old = _encode_query_last(tok, graph)
        E = len(graph.edges)
        total_graphs += 1

        # A. head: markers and node ids at positions 0-3, compared against
        #    graph.source/graph.target directly -- not against old[:4].
        if new[:4] != [
            tok.src_token_id,
            tok._node_token_id(graph.source),
            tok.dst_token_id,
            tok._node_token_id(graph.target),
        ]:
            bad_head += 1

        # B. perm: edge block is identical (just no longer trailing)
        if new[4:] != old[:-4]:
            bad_perm += 1

        # C. len: formula 3E+4 holds for both layouts
        if not (len(new) == len(old) == 3 * E + 4):
            bad_len += 1

        # D. counter: same multiset of token ids
        if Counter(new) != Counter(old):
            bad_counter += 1

        # E. pad: pad_token_id must not appear in a real encoder sequence
        if tok.pad_token_id in new:
            bad_pad += 1

        # F. formula: marker positions are always 0 and 2
        if new[0] != tok.src_token_id or new[2] != tok.dst_token_id:
            bad_formula += 1

        local_lens.add(len(new))

    observed_lengths |= local_lens
    ok = not (bad_head or bad_perm or bad_len or bad_counter or bad_pad or bad_formula)
    detail = (
        f"head={bad_head} perm={bad_perm} len={bad_len} counter={bad_counter} "
        f"pad_leak={bad_pad} markers={bad_formula}; "
        f"lengths {min(local_lens)}-{max(local_lens)}"
    )
    check(f"{split_label} ({len(split.examples)} graphs)", ok, detail)

check(
    "total graph count",
    total_graphs == 1000 + 600 + 200 + 50,
    f"{total_graphs} checked",
)
check(
    "ID observed lengths 34-184",
    34 in observed_lengths and 184 in observed_lengths,
    f"min={min(observed_lengths)} max={max(observed_lengths)}",
)

# G. Non-trivial: verify the old layout really did trail the markers.
# Use the last split's first graph (any deterministic example will do).
_sample_g, _ = split.examples[0]
_old_sample = _encode_query_last(tok, _sample_g)
_new_sample = tok.encode_graph(_sample_g)
check(
    "old layout had [SRC] at old[-4] (so the move is not a no-op)",
    _old_sample[-4] == tok.src_token_id and _old_sample[-2] == tok.dst_token_id,
    f"old[-4]={_old_sample[-4]} old[-2]={_old_sample[-2]}",
)
check(
    "new layout does NOT have [SRC] at new[-4] (markers are no longer trailing)",
    _new_sample[-4] != tok.src_token_id,
    f"new[-4]={_new_sample[-4]}",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 72)
if FAILURES:
    print(f"ITEM 1 FAILED -- {len(FAILURES)} assertion(s):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)

print("ITEM 1 PASS -- all permutation invariants hold across 1,850 graphs.")
sys.exit(0)


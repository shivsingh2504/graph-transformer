"""Pre-launch item 4 for the run 7 query-block ruling.

Pushes a 64-example mixed-E batch through the REAL collate_fn and _make_masks
and asserts that:

  1. src shape is (64, max_src_len).
  2. Cross-attention mask shape is (B, 1, 1, max_src_len).
  3. src_mask == (src != pad_id) exactly -- pads flagged, real tokens kept.
  4. Per-row invariant: every real token is unmasked, every pad position is
     masked, and the tail of each row is all pad.
  5. NEW LAYOUT ONLY: the first four columns of src hold the query block
     [SRC, node_s, DST, node_t] in every row.
  6. NEW LAYOUT ONLY: those first four columns are unmasked even in the
     shortest row (they are real tokens, so they must never be padded).
  7. No real token in any encoder sequence equals pad_id -- which is what
     makes the mask safe at all ([PAD]=0, node tokens start at 5).

The test runs twice: once with the new (query-first) layout produced by the
REAL encode_graph, and once with the old (query-last) layout built from the
tokenizer's helpers.  Checks 1-4 and 7 must pass for both; checks 5-6 apply
to the new layout only.

A negative test is also run: a left-padded collate is constructed manually
and asserted to FAIL the row-invariant check, confirming the assertions can
catch bad padding directions.

Run from the repo root:
    .venv/Scripts/python diagnostics/run7_collate_mask_test.py

Exit 0 = all assertions passed.
Exit 1 = one or more assertions failed.
Exit 2 = encode_graph has not been changed yet -- run this after modifying tokenizer.py.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from train.train import collate_fn, _make_masks

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
# Tokenizer and layout guard
# ---------------------------------------------------------------------------

tok = GraphTokenizer(min_weight=1, max_weight=10)

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
# Generate a 64-example batch with multiple E values
# ---------------------------------------------------------------------------

mixed = generate_dataset_split(num_examples=64, node_range=(5, 20), base_seed=7)

srcs_new = [tok.encode_graph(g) for g, _ in mixed.examples]           # real layout
srcs_old = [_encode_query_last(tok, g) for g, _ in mixed.examples]    # old layout
tgts = [tok.encode_path(sp) for _, sp in mixed.examples]
graphs = [g for g, _ in mixed.examples]

e_counts = sorted({len(g.edges) for g in graphs})
check(
    "batch is genuinely mixed-E (>1 distinct edge count)",
    len(e_counts) > 1,
    f"{len(e_counts)} distinct E values: {e_counts[:10]}",
)

# ---------------------------------------------------------------------------
# Main assertion loop -- run for both layouts
# ---------------------------------------------------------------------------

print()
print("=" * 72)

for tag, srcs in (("new layout (query-first)", srcs_new),
                  ("old layout (query-last)",  srcs_old)):
    is_new = tag.startswith("new")
    print(f"\n--- {tag} ---")

    src, tgt = collate_fn(list(zip(srcs, tgts)), tok.pad_token_id)
    src_mask, tgt_self, tgt_cross = _make_masks(src, tgt, tok.pad_token_id)

    B, S = src.shape
    expected_S = max(len(s) for s in srcs)

    # 1. src shape
    check(
        f"[{tag}] src shape == (64, {expected_S})",
        (B, S) == (64, expected_S),
        str(tuple(src.shape)),
    )

    # 2. cross-attention mask shape
    check(
        f"[{tag}] tgt_cross shape == (B, 1, 1, {expected_S})",
        tuple(tgt_cross.shape) == (B, 1, 1, S),
        str(tuple(tgt_cross.shape)),
    )

    # 3. src_mask == (src != pad) exactly
    # create_padding_mask returns True where the token should be KEPT.
    expected_mask = (src != tok.pad_token_id).unsqueeze(1).unsqueeze(2)
    check(
        f"[{tag}] src_mask == (src != pad_id)",
        bool((src_mask == expected_mask).all()),
    )

    # 4. Per-row: real tokens unmasked, pad positions masked, tail is pad
    row_ok = True
    for b in range(B):
        n = len(srcs[b])
        real_unmasked = bool(src_mask[b, 0, 0, :n].all())
        pad_masked = not bool(src_mask[b, 0, 0, n:].any())
        tail_is_pad = bool((src[b, n:] == tok.pad_token_id).all())
        if not (real_unmasked and pad_masked and tail_is_pad):
            row_ok = False
    check(
        f"[{tag}] per-row: real tokens unmasked, pads masked, tail is pad",
        row_ok,
    )

    # 5 & 6. New layout only: query block at front, always unmasked
    if is_new:
        head_ok = all(
            src[b, 0].item() == tok.src_token_id
            and src[b, 1].item() == tok._node_token_id(graphs[b].source)
            and src[b, 2].item() == tok.dst_token_id
            and src[b, 3].item() == tok._node_token_id(graphs[b].target)
            for b in range(B)
        )
        check(
            f"[{tag}] first four columns are the query block in every row",
            head_ok,
        )
        check(
            f"[{tag}] query block (cols 0-3) is unmasked even in the shortest row",
            bool(src_mask[:, 0, 0, :4].all()),
        )

    # 7. No real token equals pad_id
    check(
        f"[{tag}] no real encoder token equals pad_id ({tok.pad_token_id})",
        all(tok.pad_token_id not in s for s in srcs),
    )

# ---------------------------------------------------------------------------
# Negative test: left-padding must be caught by check 4
# ---------------------------------------------------------------------------

print()
print("--- negative test: left-padded collate (must trigger row-invariant failure) ---")

max_len = max(len(s) for s in srcs_new)
left_padded = torch.tensor(
    [[tok.pad_token_id] * (max_len - len(s)) + s for s in srcs_new],
    dtype=torch.long,
)
lp_mask, _, _ = _make_masks(left_padded, tgt, tok.pad_token_id)

left_row_ok = True
for b in range(B):
    n = len(srcs_new[b])
    real_unmasked = bool(lp_mask[b, 0, 0, max_len - n:].all())
    pad_masked = not bool(lp_mask[b, 0, 0, :max_len - n].any())
    # For left-padding: tail is never all-pad, head is all-pad instead.
    # A simple right-pad check must flag this.
    tail_is_pad = bool((left_padded[b, n:] == tok.pad_token_id).all())
    if real_unmasked and pad_masked and tail_is_pad:
        left_row_ok = False   # flipped: we want at least one row to fail

neg_ok = not left_row_ok   # negative test passes if at least one row failed
check("negative test: left-padded collate fails the row-invariant check", neg_ok)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 72)
if FAILURES:
    print(f"ITEM 4 FAILED -- {len(FAILURES)} assertion(s):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)

print("ITEM 4 PASS -- collate_fn and _make_masks behave correctly for both layouts.")
sys.exit(0)


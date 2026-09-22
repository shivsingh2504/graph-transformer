"""Pre-launch items 1 and 4 for the run 7 query-block ruling, executed locally.

Item 1: the reordered encoder output must be a pure permutation of the current
        one, with the query block at positions 0-3.
Item 4: a mixed-E batch through the REAL collate_fn / _make_masks must mask pads
        only, and must carry the query block in the first four columns of every
        row.

src/data/tokenizer.py is NOT modified. The new layout is built here from the
tokenizer's own helpers, so ordering is the only difference under test.
"""
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from train.train import collate_fn, _make_masks

FAILURES = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def encode_graph_query_first(tok, graph):
    """Run 7 layout: [SRC] node_s [DST] node_t, then the canonical edge block."""
    ids = [
        tok.src_token_id,
        tok._node_token_id(graph.source),
        tok.dst_token_id,
        tok._node_token_id(graph.target),
    ]
    for u, v, w in tok._canonical_edges(graph):
        ids.append(tok._node_token_id(u))
        ids.append(tok._node_token_id(v))
        ids.append(tok._weight_token_id(w))
    return ids


tok = GraphTokenizer(min_weight=1, max_weight=10)

# ---------------------------------------------------------------- item 1
print("=" * 72)
print("ITEM 1 - permutation invariants over real generated splits")
print("=" * 72)

SPLITS = [
    ("ID  N=5-20  base_seed=0", (1000, (5, 20), 0)),
    ("OOD N=21-50 base_seed=2", (600, (21, 50), 2)),
    ("fixed N=7   base_seed=3", (200, (7, 7), 3)),
    ("fixed N=50  base_seed=4", (50, (50, 50), 4)),
]

tot = 0
for label, (n, rng, seed) in SPLITS:
    split = generate_dataset_split(num_examples=n, node_range=rng, base_seed=seed)
    bad_head = bad_perm = bad_len = bad_counter = bad_pad = bad_formula = 0
    lens = set()
    for g, _sp in split.examples:
        old = tok.encode_graph(g)
        new = encode_graph_query_first(tok, g)
        tot += 1
        E = len(g.edges)

        if new[:4] != [tok.src_token_id, tok._node_token_id(g.source),
                       tok.dst_token_id, tok._node_token_id(g.target)]:
            bad_head += 1
        if new[4:] != old[:-4]:
            bad_perm += 1
        if not (len(new) == len(old) == 3 * E + 4):
            bad_len += 1
        if Counter(new) != Counter(old):
            bad_counter += 1
        if tok.pad_token_id in new:
            bad_pad += 1
        if new[0] != tok.src_token_id or new[2] != tok.dst_token_id:
            bad_formula += 1
        lens.add(len(new))

    ok = not (bad_head or bad_perm or bad_len or bad_counter or bad_pad or bad_formula)
    check(f"{label} ({len(split.examples)} graphs)", ok,
          f"head={bad_head} perm={bad_perm} len={bad_len} counter={bad_counter} "
          f"pad_leak={bad_pad} markers={bad_formula}; lengths {min(lens)}-{max(lens)}")

check("all splits covered", tot == 1000 + 600 + 200 + 50, f"{tot} graphs checked")

# The permutation must be non-trivial: confirm the old layout really did trail.
old0 = tok.encode_graph(split.examples[0][0])
new0 = encode_graph_query_first(tok, split.examples[0][0])
check("old layout had markers at the end (so the move is real)",
      old0[-4] == tok.src_token_id and old0[-2] == tok.dst_token_id
      and new0[-4] != tok.src_token_id)

# ---------------------------------------------------------------- item 4
print()
print("=" * 72)
print("ITEM 4 - mixed-E batch through the real collate_fn / _make_masks")
print("=" * 72)

mixed = generate_dataset_split(num_examples=64, node_range=(5, 20), base_seed=7)
srcs_new = [encode_graph_query_first(tok, g) for g, _ in mixed.examples]
srcs_old = [tok.encode_graph(g) for g, _ in mixed.examples]
tgts = [tok.encode_path(sp) for _, sp in mixed.examples]

e_counts = sorted({len(g.edges) for g, _ in mixed.examples})
check("batch really is mixed-E", len(e_counts) > 1,
      f"{len(e_counts)} distinct E values: {e_counts[:8]}...")

for tag, srcs in (("new layout", srcs_new), ("old layout", srcs_old)):
    src, tgt = collate_fn(list(zip(srcs, tgts)), tok.pad_token_id)
    src_mask, tgt_self, tgt_cross = _make_masks(src, tgt, tok.pad_token_id)

    B, S = src.shape
    check(f"[{tag}] src shape (B,S)", (B, S) == (64, max(map(len, srcs))),
          f"{tuple(src.shape)}")
    check(f"[{tag}] cross-attn mask shape (B,1,1,S)",
          tuple(tgt_cross.shape) == (B, 1, 1, S), f"{tuple(tgt_cross.shape)}")

    # Mask must be False exactly where the tensor holds pad, True everywhere else.
    expected_mask = (src != tok.pad_token_id).unsqueeze(1).unsqueeze(2)
    check(f"[{tag}] src_mask == (src != pad), pads only",
          bool((src_mask == expected_mask).all()))

    row_pad_ok = True
    for b in range(B):
        n = len(srcs[b])
        if not (bool(src_mask[b, 0, 0, :n].all())
                and not bool(src_mask[b, 0, 0, n:].any())
                and bool((src[b, n:] == tok.pad_token_id).all())):
            row_pad_ok = False
    check(f"[{tag}] every row: real tokens unmasked, pads masked, tail is pad",
          row_pad_ok)

    head_ok = all(
        src[b, 0].item() == tok.src_token_id
        and src[b, 1].item() == tok._node_token_id(mixed.examples[b][0].source)
        and src[b, 2].item() == tok.dst_token_id
        and src[b, 3].item() == tok._node_token_id(mixed.examples[b][0].target)
        for b in range(B)
    ) if tag == "new layout" else True
    if tag == "new layout":
        check("[new layout] first four columns are the query block in every row",
              head_ok)
        check("[new layout] query block unmasked even in the shortest row",
              bool(src_mask[:, 0, 0, :4].all()))

    # No real token may collide with pad_id, or the mask would eat content.
    check(f"[{tag}] no real token equals pad_id",
          all(tok.pad_token_id not in s for s in srcs))

print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILURES:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("items 1 and 4 PASS - no failures")
sys.exit(0)

"""Negative tests for diagnostics/run7_layout_probe.py.

Four deliberately broken variants. If the item 1 / item 4 assertions are real,
each must be caught. A green probe run means nothing unless these fail.
Reuses the same assertion logic, pointed at broken code.
"""
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from train.train import _make_masks

tok = GraphTokenizer(min_weight=1, max_weight=10)
split = generate_dataset_split(num_examples=200, node_range=(5, 20), base_seed=0)
ex = split.examples


def good(g):
    ids = [tok.src_token_id, tok._node_token_id(g.source),
           tok.dst_token_id, tok._node_token_id(g.target)]
    for u, v, w in tok._canonical_edges(g):
        ids += [tok._node_token_id(u), tok._node_token_id(v), tok._weight_token_id(w)]
    return ids


def dropped_token(g):
    """Bug: forgets the [DST] marker, so length and multiset both change."""
    ids = [tok.src_token_id, tok._node_token_id(g.source),
           tok._node_token_id(g.target)]
    for u, v, w in tok._canonical_edges(g):
        ids += [tok._node_token_id(u), tok._node_token_id(v), tok._weight_token_id(w)]
    return ids


def swapped_markers(g):
    """Bug: emits [DST] t [SRC] s - same multiset, wrong head order."""
    ids = [tok.dst_token_id, tok._node_token_id(g.target),
           tok.src_token_id, tok._node_token_id(g.source)]
    for u, v, w in tok._canonical_edges(g):
        ids += [tok._node_token_id(u), tok._node_token_id(v), tok._weight_token_id(w)]
    return ids


def item1_counts(fn):
    head = perm = ln = ctr = 0
    for g, _ in ex:
        old = tok.encode_graph(g)
        new = fn(g)
        E = len(g.edges)
        if new[:4] != [tok.src_token_id, tok._node_token_id(g.source),
                       tok.dst_token_id, tok._node_token_id(g.target)]:
            head += 1
        if new[4:] != old[:-4]:
            perm += 1
        if not (len(new) == len(old) == 3 * E + 4):
            ln += 1
        if Counter(new) != Counter(old):
            ctr += 1
    return head, perm, ln, ctr


print("=" * 72)
print("A. dropped [DST] marker - expect head/perm/len/counter all non-zero")
h, p, l, c = item1_counts(dropped_token)
print(f"   head={h} perm={p} len={l} counter={c}")
print(f"   {'CAUGHT' if (h and p and l and c) else 'NOT CAUGHT <-- assertion is vacuous'}")

print()
print("B. swapped [SRC]/[DST] order - FINDING, not a failure")
h, p, l, c = item1_counts(swapped_markers)
print(f"   head={h} perm={p} len={l} counter={c}")
print(f"   {'head check CAUGHT it' if h else 'NOT CAUGHT <-- assertion is vacuous'}")
print("   perm/len/counter are all 0 because new[4:] == old[:-4] compares only")
print("   the EDGE block: a bug confined to the query block is invisible to it,")
print("   and Counter is order-blind by construction. So the explicit head")
print("   assertion is load-bearing and cannot be dropped from item 1.")

print()
print("D. query nodes swapped, markers correct ([SRC] target [DST] source)")
print("   - the dangerous one: it asks the model the wrong question while")
print("     every structural check still passes")


def swapped_nodes(g):
    ids = [tok.src_token_id, tok._node_token_id(g.target),
           tok.dst_token_id, tok._node_token_id(g.source)]
    for u, v, w in tok._canonical_edges(g):
        ids += [tok._node_token_id(u), tok._node_token_id(v), tok._weight_token_id(w)]
    return ids


h, p, l, c = item1_counts(swapped_nodes)
print(f"   head={h} perm={p} len={l} counter={c}")
n_swappable = sum(1 for g, _ in ex if g.source != g.target)
print(f"   graphs where source != target (so the swap is observable): "
      f"{n_swappable}/{len(ex)}")
caught_by_head_only = bool(h == n_swappable) and not (p or l or c)
print(f"   {'CAUGHT by the head check ALONE - perm/len/counter all blind' if caught_by_head_only else 'UNEXPECTED <-- investigate'}")

print()
print("=" * 72)
print("C. left-padding collate - expect the row/mask and head checks to fail")


def left_pad_collate(batch, pad_id):
    """Bug: pads on the left, which silently shifts every absolute position."""
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    src = torch.tensor([[pad_id] * (max_src - len(s)) + list(s) for s in srcs],
                       dtype=torch.long)
    tgt = torch.tensor([list(t) + [pad_id] * (max_tgt - len(t)) for t in tgts],
                       dtype=torch.long)
    return src, tgt


srcs_new = [good(g) for g, _ in ex[:64]]
tgts = [tok.encode_path(sp) for _, sp in ex[:64]]
src, tgt = left_pad_collate(list(zip(srcs_new, tgts)), tok.pad_token_id)
src_mask, _, _ = _make_masks(src, tgt, tok.pad_token_id)
B, S = src.shape

tail_is_pad = all(bool((src[b, len(srcs_new[b]):] == tok.pad_token_id).all())
                  for b in range(B))
head_is_query = all(src[b, 0].item() == tok.src_token_id for b in range(B))
shortest_unmasked = bool(src_mask[:, 0, 0, :4].all())

print(f"   tail-is-pad (right-pad assumption): {tail_is_pad}")
print(f"   first four columns are query block : {head_is_query}")
print(f"   query block unmasked in every row  : {shortest_unmasked}")
print(f"   {'CAUGHT' if not (tail_is_pad and head_is_query and shortest_unmasked) else 'NOT CAUGHT <-- assertion is vacuous'}")

"""
diagnostic.py – read-only post-run analysis for the M8 baseline.

Answers three questions:
  1. Is the model undertrained or does it fail to generalise?
     -> evaluate on 1 000 examples drawn from the TRAINING distribution (seed 0)
        and compare train-set accuracy with the ID eval accuracy (0.629).
  2. Where do ID failures cluster?
     -> break down by graph node count and by true shortest-path length.
  3. Why do outputs fail?
     -> classify each failure as:
         NON_EDGE_HOP   - a hop that is not an edge in the graph
         REPEATED_NODE  - a node appears more than once (cycle)
         PREMATURE_EOS  - EOS before reaching target (but edge-valid so far)
         WRONG_ENDPOINT - edges OK, no repeat, but wrong start or end node
         NO_EOS         - hit the decode limit without emitting EOS
         SUBOPTIMAL     - valid simple path with correct endpoints but non-optimal cost

Usage (run from the repo root):
    python src/diagnostic.py
"""

from __future__ import annotations

import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# path bootstrap so the script can be run from the repo root or from src/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from data.graph_generator import Graph
from data.dijkstra import ShortestPath
from model.model import Transformer
from eval.evaluate import _greedy_decode, _MAX_DECODE_LEN  # reuse helpers

# ---------------------------------------------------------------------------
# constants - must match the training run
# ---------------------------------------------------------------------------
_NODE_RANGE = (5, 20)
_N_TRAIN_DIAG = 1_000   # examples drawn from training distribution
_N_ID_DIAG    = 1_000   # must equal num_eval_examples in the run
_TRAIN_SEED   = 0
_EVAL_SEED    = 1

# ---------------------------------------------------------------------------
# failure-mode constants
# ---------------------------------------------------------------------------
NON_EDGE_HOP    = "NON_EDGE_HOP"
REPEATED_NODE   = "REPEATED_NODE"
PREMATURE_EOS   = "PREMATURE_EOS"
WRONG_ENDPOINT  = "WRONG_ENDPOINT"
NO_EOS          = "NO_EOS"
SUBOPTIMAL      = "SUBOPTIMAL"

FailureMode = str


def _classify_failure(
    raw_tokens: List[int],
    graph: Graph,
    tokenizer: GraphTokenizer,
) -> FailureMode:
    """Classify one failing example.  Priority: first bad thing encountered."""
    # did we get an EOS at all?
    if tokenizer.eos_token_id not in raw_tokens[1:]:
        return NO_EOS

    # extract node list up to EOS
    nodes: List[int] = []
    for tid in raw_tokens[1:]:
        if tid == tokenizer.eos_token_id:
            break
        tok = tokenizer.id_to_token.get(tid, "")
        if tok.startswith("node_"):
            try:
                nodes.append(int(tok[len("node_"):]))
            except ValueError:
                continue  # malformed node_ token; skip (not silenced)

    # repeated node (cycle)
    if len(set(nodes)) != len(nodes):
        return REPEATED_NODE

    edge_set: Set[Tuple[int, int]] = {
        (min(u, v), max(u, v)) for u, v, _ in graph.edges
    }

    # walk hops looking for a bad edge
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        if (min(u, v), max(u, v)) not in edge_set:
            return NON_EDGE_HOP

    # edges all valid - check endpoints
    if len(nodes) == 0 or nodes[-1] != graph.target:
        if len(nodes) > 0 and nodes[0] == graph.source:
            return PREMATURE_EOS
        return WRONG_ENDPOINT

    if nodes[0] != graph.source:
        return WRONG_ENDPOINT

    # valid simple path with correct endpoints -> suboptimal
    return SUBOPTIMAL


# ---------------------------------------------------------------------------
# lean per-example evaluation
# ---------------------------------------------------------------------------

def _eval_one(
    model: Transformer,
    graph: Graph,
    sp: ShortestPath,
    tokenizer: GraphTokenizer,
    device: torch.device,
) -> Tuple[bool, Optional[FailureMode]]:
    """Returns (valid_and_optimal, failure_mode_or_None)."""
    src_ids = torch.tensor(
        [tokenizer.encode_graph(graph)], dtype=torch.long
    )

    raw_tokens = _greedy_decode(model, src_ids, tokenizer, device, _MAX_DECODE_LEN)

    # decode node list
    nodes: List[int] = []
    has_eos = False
    for tid in raw_tokens[1:]:
        if tid == tokenizer.eos_token_id:
            has_eos = True
            break
        tok = tokenizer.id_to_token.get(tid, "")
        if tok.startswith("node_"):
            try:
                nodes.append(int(tok[len("node_"):]))
            except ValueError:
                continue  # malformed node_ token; skip (not silenced)

    if not has_eos:
        return False, NO_EOS

    edge_set: Set[Tuple[int, int]] = {
        (min(u, v), max(u, v)) for u, v, _ in graph.edges
    }
    adj: Dict[int, List[Tuple[int, int]]] = graph.adjacency()

    edges_valid = True
    cost = 0
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        canonical = (min(u, v), max(u, v))
        if canonical not in edge_set:
            edges_valid = False
            break
        for nb, w in adj[u]:
            if nb == v:
                cost += w
                break
        else:
            raise AssertionError(
                f"Edge ({u},{v}) passed edge_set check but not found in adj[u]"
            )

    is_simple  = len(set(nodes)) == len(nodes)
    correct_ep = len(nodes) > 0 and nodes[0] == graph.source and nodes[-1] == graph.target
    optimal    = edges_valid and is_simple and correct_ep and cost == sp.cost

    if optimal:
        return True, None

    mode = _classify_failure(raw_tokens, graph, tokenizer)
    return False, mode


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to final.pt, e.g. checkpoints_run5/final.pt")
    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = ckpt_path.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg  = ckpt["model_config"]
    tok_kwargs = ckpt["tokenizer_kwargs"]

    tokenizer = GraphTokenizer(**tok_kwargs)
    model = Transformer(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 1. Train-distribution sample
    # ------------------------------------------------------------------
    print(f"\nGenerating {_N_TRAIN_DIAG} train-dist examples (seed={_TRAIN_SEED}) ...")
    train_split = generate_dataset_split(
        num_examples=_N_TRAIN_DIAG,
        node_range=_NODE_RANGE,
        base_seed=_TRAIN_SEED,
    )
    train_correct = 0
    for i, (graph, sp) in enumerate(train_split.examples):
        ok, _ = _eval_one(model, graph, sp, tokenizer, device)
        if ok:
            train_correct += 1
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{_N_TRAIN_DIAG}")

    train_acc = train_correct / _N_TRAIN_DIAG
    print(f"Train-dist accuracy : {train_acc:.3f}  ({train_correct}/{_N_TRAIN_DIAG})")

    # ------------------------------------------------------------------
    # 2. ID eval split
    # ------------------------------------------------------------------
    print(f"\nGenerating {_N_ID_DIAG} ID eval examples (seed={_EVAL_SEED}) ...")
    id_split = generate_dataset_split(
        num_examples=_N_ID_DIAG,
        node_range=_NODE_RANGE,
        base_seed=_EVAL_SEED,
    )

    id_correct = 0
    failures_by_nodes:    Dict[int, int] = defaultdict(int)
    failures_by_path_len: Dict[int, int] = defaultdict(int)
    totals_by_nodes:      Dict[int, int] = defaultdict(int)
    totals_by_path_len:   Dict[int, int] = defaultdict(int)
    failure_modes:        Dict[str, int] = defaultdict(int)

    for i, (graph, sp) in enumerate(id_split.examples):
        n  = graph.num_nodes
        pl = len(sp.path)
        totals_by_nodes[n]    += 1
        totals_by_path_len[pl] += 1

        ok, mode = _eval_one(model, graph, sp, tokenizer, device)
        if ok:
            id_correct += 1
        else:
            failures_by_nodes[n]    += 1
            failures_by_path_len[pl] += 1
            failure_modes[mode]      += 1  # type: ignore[index]

        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{_N_ID_DIAG}")

    id_acc = id_correct / _N_ID_DIAG
    print(f"ID eval accuracy    : {id_acc:.3f}  ({id_correct}/{_N_ID_DIAG})")

    # ------------------------------------------------------------------
    # 3. Verdict
    # ------------------------------------------------------------------
    gap = train_acc - id_acc
    print("\n" + "=" * 60)
    print("UNDERTRAINING vs GENERALISATION")
    print("=" * 60)
    print(f"  train-dist acc : {train_acc:.3f}")
    print(f"  ID eval acc    : {id_acc:.3f}")
    print(f"  gap            : {gap:+.3f}")
    if train_acc < 0.70:
        verdict = "UNDERTRAINED (train-set accuracy also low)"
    elif gap > 0.15:
        verdict = "GENERALISATION GAP (fits training dist but fails on eval)"
    else:
        verdict = "MIXED (modest gap; both effects may contribute)"
    print(f"  verdict        : {verdict}")

    # ------------------------------------------------------------------
    # 4. Breakdown by node count
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ID FAILURES BY NODE COUNT")
    print("=" * 60)
    print(f"  {'nodes':>6}  {'total':>6}  {'fail':>5}  {'fail%':>6}  {'acc%':>6}")
    for n in sorted(totals_by_nodes):
        tot  = totals_by_nodes[n]
        fail = failures_by_nodes.get(n, 0)
        print(f"  {n:>6}  {tot:>6}  {fail:>5}  {100*fail/tot:>5.1f}%  {100*(tot-fail)/tot:>5.1f}%")

    # ------------------------------------------------------------------
    # 5. Breakdown by path length
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ID FAILURES BY TRUE PATH LENGTH (# nodes in path)")
    print("=" * 60)
    print(f"  {'path_len':>8}  {'total':>6}  {'fail':>5}  {'fail%':>6}  {'acc%':>6}")
    for pl in sorted(totals_by_path_len):
        tot  = totals_by_path_len[pl]
        fail = failures_by_path_len.get(pl, 0)
        print(f"  {pl:>8}  {tot:>6}  {fail:>5}  {100*fail/tot:>5.1f}%  {100*(tot-fail)/tot:>5.1f}%")

    # ------------------------------------------------------------------
    # 6. Failure mode classification
    # ------------------------------------------------------------------
    total_failures = _N_ID_DIAG - id_correct
    print("\n" + "=" * 60)
    print("FAILURE MODE CLASSIFICATION  (ID eval)")
    print("=" * 60)
    mode_order = [NON_EDGE_HOP, REPEATED_NODE, PREMATURE_EOS, WRONG_ENDPOINT, NO_EOS, SUBOPTIMAL]
    for m in mode_order:
        cnt = failure_modes.get(m, 0)
        pct = 100 * cnt / total_failures if total_failures else 0.0
        print(f"  {m:<20}  {cnt:>4}  ({pct:5.1f}% of failures)")
    unaccounted = total_failures - sum(failure_modes.get(m, 0) for m in mode_order)
    if unaccounted:
        print(f"  {'OTHER':<20}  {unaccounted:>4}  ({100*unaccounted/total_failures:5.1f}% of failures)")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out = {
        "train_dist_accuracy": train_acc,
        "id_eval_accuracy": id_acc,
        "gap": gap,
        "verdict": verdict,
        "failures_by_node_count": {
            str(n): {"total": totals_by_nodes[n], "failures": failures_by_nodes.get(n, 0)}
            for n in sorted(totals_by_nodes)
        },
        "failures_by_path_length": {
            str(pl): {"total": totals_by_path_len[pl], "failures": failures_by_path_len.get(pl, 0)}
            for pl in sorted(totals_by_path_len)
        },
        "failure_modes": dict(failure_modes),
    }
    out_path = out_dir / "diagnostic.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()

"""
Breakdown analysis of run5 eval results -- v2 (bugs fixed, moved from scratch/).

Fixes vs v1:
  - Adds `has_cycle` bucket: correct endpoints + all-valid edges + repeated node.
    These are NOT valid paths (is_simple=False) and were wrongly counted as
    suboptimal in v1.
  - Reports two weight-sensitivity definitions side by side; Def A (data_check.py)
    is the locked primary number.
  - All three sanity counts cross-checked against results.json.

Key correction vs v1 label:
  Cycles are structural failures (like invalid_edge), not cost mistakes.
  True "valid path but suboptimal" = fail_valid_suboptimal (113), not 129.
  The 116 from results.json = 113 (bucket 4) + 3 (wrong-endpoint but valid path).
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from heapq import heappush, heappop

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
sys.path.insert(0, _SRC)

import torch

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import evaluate_example, ExampleResult
from model.model import Transformer

# ---------------------------------------------------------------------------
# Config -- matches run_final_training.RunConfig exactly
# ---------------------------------------------------------------------------
EVAL_SEED = 1
NUM_EVAL = 1_000
NODE_RANGE = (5, 20)
CKPT_PATH = os.path.join(_HERE, "..", "checkpoints_run5", "final.pt")


def _bfs_min_hops(graph, source, target) -> int:
    """Minimum hop count from source to target (unweighted BFS)."""
    adj = graph.adjacency()
    visited = {source: 0}
    q = deque([(source, 0)])
    while q:
        u, d = q.popleft()
        if u == target:
            return d
        for v, _ in adj[u]:
            if v not in visited:
                visited[v] = d + 1
                q.append((v, d + 1))
    return float("inf")


def _weight_sensitive_datacheck(graph, sp) -> bool:
    """
    Definition A -- matches data_check.py (locked primary number, see lock.md).
    Sensitive = hop_count(Dijkstra path) != BFS min hops.
    i.e. Dijkstra took a longer path in hops because the hop-min route was
    more expensive in weight.

    Caveat: uses whichever optimal path Dijkstra returns; ties are common
    with integer weights in [1, 10].
    """
    min_hop = _bfs_min_hops(graph, graph.source, graph.target)
    dijkstra_hops = len(sp.path) - 1
    return dijkstra_hops != min_hop


def _weight_sensitive_costbfs(graph, sp) -> bool:
    """
    Definition B -- v1 BFS-cost (wider; shown for comparison only).
    Sensitive = cost(BFS-min-hop path) > Dijkstra cost.
    Catches cases where a same-hop path exists but is cheaper, which
    Def A misses because Dijkstra's hop count equals BFS.
    """
    adj = graph.adjacency()
    INF = float("inf")
    hop_dist = {n: INF for n in graph.node_ids}
    hop_dist[graph.source] = 0
    prev: dict = {graph.source: None}
    pq = [(0, graph.source)]
    while pq:
        d, u = heappop(pq)
        if d > hop_dist[u]:
            continue
        for v, _ in adj[u]:
            nd = d + 1
            if nd < hop_dist[v]:
                hop_dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))

    if hop_dist[graph.target] == INF:
        return False

    path = []
    cur = graph.target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for nbr, w in adj[u]:
            if nbr == v:
                cost += w
                break

    return cost > sp.cost


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(CKPT_PATH):
        print(f"\nERROR: checkpoint not found at {CKPT_PATH}")
        sys.exit(1)

    print(f"Loading checkpoint from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
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
    print("Model loaded.\n")

    print(f"Generating {NUM_EVAL} eval graphs (seed={EVAL_SEED}, nodes={NODE_RANGE}) ...")
    eval_split = generate_dataset_split(
        num_examples=NUM_EVAL,
        node_range=NODE_RANGE,
        base_seed=EVAL_SEED,
    )
    print("Eval split ready.\n")

    BUCKETS = [(5, 9), (10, 14), (15, 20)]

    bucket_total = {b: 0 for b in BUCKETS}
    bucket_vo = {b: 0 for b in BUCKETS}

    # Taxonomy -- checked in this order; first failing condition wins.
    # Consequence: a graph with wrong endpoints AND invalid edges appears only
    # in fail_wrong_endpoint. So fail_invalid_edge < raw edges_valid=False count.
    fail_wrong_endpoint = 0    # correct_endpoints = False
    fail_invalid_edge = 0      # correct_endpoints = True,  edges_valid = False
    fail_has_cycle = 0         # correct_endpoints = True,  edges_valid = True,
                               #   valid_path = False  (repeated node, not simple)
    fail_valid_suboptimal = 0  # correct_endpoints = True,  valid_path = True,
                               #   optimal_cost = False

    # Raw sanity counters (not filtered by prior conditions)
    _count_edges_invalid = 0
    _count_valid_path = 0

    # Weight sensitivity
    ws_dc_total = 0; ws_dc_vo = 0     # Def A: data_check.py (locked)
    nws_dc_total = 0; nws_dc_vo = 0
    ws_v1_total = 0; ws_v1_vo = 0     # Def B: BFS-cost (comparison only)
    nws_v1_total = 0; nws_v1_vo = 0

    t0 = time.time()
    for idx, (graph, sp) in enumerate(eval_split.examples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{NUM_EVAL}  ({time.time()-t0:.1f}s elapsed) ...")

        result: ExampleResult = evaluate_example(model, graph, sp, tokenizer, device)
        n = graph.num_nodes

        for b in BUCKETS:
            if b[0] <= n <= b[1]:
                bucket_total[b] += 1
                if result.valid_and_optimal:
                    bucket_vo[b] += 1
                break

        if not result.edges_valid:
            _count_edges_invalid += 1
        if result.valid_path:
            _count_valid_path += 1

        if not result.valid_and_optimal:
            if not result.correct_endpoints:
                fail_wrong_endpoint += 1
            elif not result.edges_valid:
                fail_invalid_edge += 1
            elif not result.valid_path:
                fail_has_cycle += 1
            else:
                fail_valid_suboptimal += 1

        dc = _weight_sensitive_datacheck(graph, sp)
        if dc:
            ws_dc_total += 1
            if result.valid_and_optimal: ws_dc_vo += 1
        else:
            nws_dc_total += 1
            if result.valid_and_optimal: nws_dc_vo += 1

        v1 = _weight_sensitive_costbfs(graph, sp)
        if v1:
            ws_v1_total += 1
            if result.valid_and_optimal: ws_v1_vo += 1
        else:
            nws_v1_total += 1
            if result.valid_and_optimal: nws_v1_vo += 1

    elapsed_total = time.time() - t0
    print(f"\nEvaluation complete in {elapsed_total:.1f}s.\n")

    total = NUM_EVAL
    vo_total = sum(bucket_vo[b] for b in BUCKETS)
    n_fail = total - vo_total

    print("=" * 62)
    print("RUN 5 -- BREAKDOWN ANALYSIS (v2)")
    print("=" * 62)

    print("\n--- Sanity check vs results.json ---")
    print(f"  valid_and_optimal :  {vo_total:4d} / {total}  "
          f"(results.json: 836)   {'OK' if vo_total==836 else 'MISMATCH'}")
    print(f"  edges_valid=False :  {_count_edges_invalid:4d} / {total}  "
          f"(results.json: 30)    {'OK' if _count_edges_invalid==30 else 'MISMATCH'}")
    print(f"  valid_path=True   :  {_count_valid_path:4d} / {total}  "
          f"(results.json: 952)   {'OK' if _count_valid_path==952 else 'MISMATCH'}")

    print("\n--- 1. valid_and_optimal by node-count bucket ---")
    print(f"  {'Bucket':<8} {'N':>6} {'V&O':>6} {'%':>8}")
    print("  " + "-" * 32)
    for b in BUCKETS:
        n = bucket_total[b]
        vo = bucket_vo[b]
        pct = 100.0 * vo / n if n > 0 else 0.0
        print(f"  {b[0]:2d}-{b[1]:2d}    {n:>6} {vo:>6} {pct:>7.1f}%")
    print("  " + "-" * 32)
    print(f"  ALL      {total:>6} {vo_total:>6} {100.0*vo_total/total:>7.1f}%")

    print(f"\n--- 2. Failure taxonomy ({n_fail} failures) ---")
    print("  Endpoint checked before edge validity; a graph with both wrong")
    print("  endpoints and invalid edges appears only in 'wrong endpoint'.")
    print("  Cycles (has_cycle) are structural failures, not cost mistakes.")

    def pf(x: int) -> str:
        return (f"{x:4d}  ({100.0*x/n_fail:.1f}% of failures, "
                f"{100.0*x/total:.1f}% of total)") if n_fail else "n/a"

    struct = fail_wrong_endpoint + fail_invalid_edge + fail_has_cycle
    cost   = fail_valid_suboptimal

    print(f"  Wrong endpoint:        {pf(fail_wrong_endpoint)}")
    print(f"  Invalid edge:          {pf(fail_invalid_edge)}")
    print(f"  Has cycle:             {pf(fail_has_cycle)}")
    print(f"  Valid but suboptimal:  {pf(fail_valid_suboptimal)}")
    print(f"  --- structural total:  {struct:4d}  ({100.0*struct/n_fail:.1f}% of failures)")
    print(f"  --- cost-error total:  {cost:4d}  ({100.0*cost/n_fail:.1f}% of failures)")
    print(f"  Cross-check: valid_path=True but not optimal = "
          f"{_count_valid_path - vo_total}  "
          f"(= {fail_valid_suboptimal} suboptimal + "
          f"{_count_valid_path - vo_total - fail_valid_suboptimal} wrong-endpoint-but-valid)")

    print("\n--- 3. Weight sensitivity ---")
    print("  'Insensitive' means the cheapest path is also a minimum-hop path.")
    print("  A non-optimal answer on an insensitive graph is still an error;")
    print("  it just happens on a graph where a longer valid route costs more.")
    print()
    print("  Def A -- data_check.py (locked primary, see lock.md):")
    print("    Sensitive = hop_count(Dijkstra path) != BFS min hops")
    dc_pct = 100.0 * ws_dc_total / total
    dc_acc_s = 100.0 * ws_dc_vo / ws_dc_total if ws_dc_total else float("nan")
    dc_acc_n = 100.0 * nws_dc_vo / nws_dc_total if nws_dc_total else float("nan")
    print(f"    Sensitive:         {ws_dc_total:4d} / {total}  ({dc_pct:.1f}%)  "
          f"[train split: 36.7%]")
    print(f"    Acc, sensitive:     {ws_dc_vo:4d} / {ws_dc_total}  = {dc_acc_s:.1f}%")
    print(f"    Acc, insensitive:   {nws_dc_vo:4d} / {nws_dc_total}  = {dc_acc_n:.1f}%")
    print(f"    Gap: {dc_acc_n - dc_acc_s:.1f} pp")
    print()
    print("  Def B -- BFS-cost (v1; comparison only, not locked):")
    print("    Sensitive = cost(BFS-min-hop path) > Dijkstra cost")
    v1_pct = 100.0 * ws_v1_total / total
    v1_acc_s = 100.0 * ws_v1_vo / ws_v1_total if ws_v1_total else float("nan")
    v1_acc_n = 100.0 * nws_v1_vo / nws_v1_total if nws_v1_total else float("nan")
    print(f"    Sensitive:         {ws_v1_total:4d} / {total}  ({v1_pct:.1f}%)")
    print(f"    Acc, sensitive:     {ws_v1_vo:4d} / {ws_v1_total}  = {v1_acc_s:.1f}%")
    print(f"    Acc, insensitive:   {nws_v1_vo:4d} / {nws_v1_total}  = {v1_acc_n:.1f}%")
    print(f"    Gap: {v1_acc_n - v1_acc_s:.1f} pp")

    print("\n" + "=" * 62)
    print("Raw counts:")
    print(f"  bucket_total = {dict(bucket_total)}")
    print(f"  bucket_vo    = {dict(bucket_vo)}")
    print(f"  fail_wrong_endpoint   = {fail_wrong_endpoint}")
    print(f"  fail_invalid_edge     = {fail_invalid_edge}")
    print(f"  fail_has_cycle        = {fail_has_cycle}")
    print(f"  fail_valid_suboptimal = {fail_valid_suboptimal}")
    print(f"  [Def A] ws={ws_dc_total} ws_vo={ws_dc_vo}  nws={nws_dc_total} nws_vo={nws_dc_vo}")
    print(f"  [Def B] ws={ws_v1_total} ws_vo={ws_v1_vo}  nws={nws_v1_total} nws_vo={nws_v1_vo}")
    print("=" * 62)


if __name__ == "__main__":
    main()

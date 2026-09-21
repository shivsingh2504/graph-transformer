from __future__ import annotations
import os
import sys
import time
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import evaluate_example
from model.model import Transformer

CKPT_PATH = os.path.join(_HERE, "..", "checkpoints_run5", "final.pt")


def _bfs_min_hops(graph, source, target) -> int:
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


def _weight_sensitive(graph, sp) -> bool:
    return (len(sp.path) - 1) != _bfs_min_hops(graph, graph.source, graph.target)


def print_waiver_note():
    print("=" * 70)
    print("M9 OOD EVALUATION -- checkpoints_run5\\final.pt")
    print("=" * 70)
    print("WAIVER NOTE:")
    print("Run 5 ID Eval Score: 83.6% (836/1000) valid_and_optimal.")
    print("Gate (>=90%) waived by project owner so OOD evaluation can proceed.")
    print("ID accuracy by size: 5-9: 95.1%, 10-14: 82.5%, 15-20: 75.7%.")
    print("=" * 70)
    print()


def main():
    print_waiver_note()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        sys.exit(1)
        
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

    print("--- CONTROL STEP ---")
    id_split = generate_dataset_split(num_examples=1000, node_range=(5, 20), base_seed=1)
    vo_count = 0
    t0_ctrl = time.time()
    for graph, sp in id_split.examples:
        res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)
        if res.valid_and_optimal:
            vo_count += 1
            
    print(f"Control ID eval valid_and_optimal: {vo_count}/1000 (in {time.time() - t0_ctrl:.1f}s)")
    if vo_count != 836:
        print(f"ERROR: Control step failed. Expected 836, got {vo_count}.")
        sys.exit(1)
    print("Control step passed.\n")
    
    print("--- OOD EVALUATION ---")
    ood_split = generate_dataset_split(num_examples=3000, node_range=(25, 50), base_seed=2)
    
    bins = [(25, 29), (30, 34), (35, 39), (40, 44), (45, 50)]
    stats = {b: {
        "n": 0, "vo": 0, "edges_valid": 0, "valid_path": 0, "correct_end": 0, "opt_cost": 0,
        "fail_wrong_end": 0, "fail_invalid_edge": 0, "fail_cycle": 0, "fail_subopt": 0,
        "ws_total": 0, "ws_vo": 0
    } for b in bins}
    
    overall = {
        "n": 0, "vo": 0, "edges_valid": 0, "valid_path": 0, "correct_end": 0, "opt_cost": 0,
        "fail_wrong_end": 0, "fail_invalid_edge": 0, "fail_cycle": 0, "fail_subopt": 0,
        "ws_total": 0, "ws_vo": 0
    }
    
    t0 = time.time()
    for idx, (graph, sp) in enumerate(ood_split.examples):
        if (idx+1) % 100 == 0:
            print(f"  {idx+1}/3000 ({time.time() - t0:.1f}s elapsed) ...")
        res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)
        
        n = graph.num_nodes
        b_key = None
        for b in bins:
            if b[0] <= n <= b[1]:
                b_key = b
                break
                
        is_ws = _weight_sensitive(graph, sp)
        
        for d in [stats[b_key], overall]:
            d["n"] += 1
            if res.valid_and_optimal: d["vo"] += 1
            if res.edges_valid: d["edges_valid"] += 1
            if res.valid_path: d["valid_path"] += 1
            if res.correct_endpoints: d["correct_end"] += 1
            if res.optimal_cost: d["opt_cost"] += 1
            
            if is_ws:
                d["ws_total"] += 1
                if res.valid_and_optimal:
                    d["ws_vo"] += 1
                    
            if not res.valid_and_optimal:
                if not res.correct_endpoints:
                    d["fail_wrong_end"] += 1
                elif not res.edges_valid:
                    d["fail_invalid_edge"] += 1
                elif not res.valid_path:
                    d["fail_cycle"] += 1
                else:
                    d["fail_subopt"] += 1

    print(f"\nOOD Evaluation complete in {time.time() - t0:.1f}s.\n")
    
    def print_report(name, d):
        count = d["n"]
        if count == 0: return
        print(f"[{name}] N={count}")
        print("  Metrics:")
        print(f"    valid_and_optimal : {d['vo']:4d} ({100.0*d['vo']/count:>5.1f}%)")
        print(f"    edges_valid       : {d['edges_valid']:4d} ({100.0*d['edges_valid']/count:>5.1f}%)")
        print(f"    valid_path        : {d['valid_path']:4d} ({100.0*d['valid_path']/count:>5.1f}%)")
        print(f"    correct_endpoints : {d['correct_end']:4d} ({100.0*d['correct_end']/count:>5.1f}%)")
        print(f"    optimal_cost      : {d['opt_cost']:4d} ({100.0*d['opt_cost']/count:>5.1f}%)")
        n_fail = count - d["vo"]
        print("  Taxonomy (exclusive, checked in this order):")
        def pf(x): return f"{x:4d} ({100.0*x/n_fail:>5.1f}% of fails)" if n_fail else "0"
        print(f"    Wrong endpoint    : {pf(d['fail_wrong_end'])}")
        print(f"    Invalid edge      : {pf(d['fail_invalid_edge'])}")
        print(f"    Has cycle         : {pf(d['fail_cycle'])}")
        print(f"    Suboptimal        : {pf(d['fail_subopt'])}")
        print("  Weight-sensitive:")
        ws_t = d["ws_total"]
        print(f"    Share             : {ws_t:4d} ({100.0*ws_t/count:>5.1f}%)")
        print(f"    Acc (sensitive)   : {d['ws_vo']:4d} / {ws_t} ({100.0*d['ws_vo']/ws_t if ws_t else 0:>5.1f}%)")
        print("-" * 55)
        
    print_report("OVERALL 25-50", overall)
    for b in bins:
        print_report(f"BIN {b[0]}-{b[1]}", stats[b])

if __name__ == "__main__":
    main()

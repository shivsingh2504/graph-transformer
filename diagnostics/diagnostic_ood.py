import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import torch
from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import evaluate_example, _greedy_decode, _decode_token_sequence
from model.model import Transformer

CKPT_PATH = os.path.join(_HERE, "..", "checkpoints_run5", "final.pt")

def bin_for(n):
    if 5 <= n <= 9: return (5,9)
    if 10 <= n <= 14: return (10,14)
    if 15 <= n <= 20: return (15,20)
    if 25 <= n <= 29: return (25,29)
    if 30 <= n <= 34: return (30,34)
    if 35 <= n <= 39: return (35,39)
    if 40 <= n <= 44: return (40,44)
    if 45 <= n <= 50: return (45,50)
    return None

def main():
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

    def run_eval(split_name, split):
        print(f"\n=======================================================")
        print(f" {split_name} DIAGNOSTICS")
        print(f"=======================================================")
        
        bins = [(5,9), (10,14), (15,20)] if "ID" in split_name else [(25,29), (30,34), (35,39), (40,44), (45,50)]
        stats = {b: {
            "n": 0, "path0_correct": 0, "eos_ended": 0, "max_ended": 0,
            "gen_lengths": [], "dj_lengths": [],
            "edges_valid": 0, "simple_path": 0, "ends_at_target": 0, "all_three": 0,
            "last_adj_target": 0
        } for b in bins}
        
        examples_printed = 0
        t0 = time.time()
        for idx, (graph, sp) in enumerate(split.examples):
            if (idx+1) % 100 == 0:
                print(f"  {idx+1}/{len(split.examples)} ({time.time()-t0:.1f}s) ...")
                
            n = graph.num_nodes
            b_key = bin_for(n)
            if b_key not in stats: continue
            
            src_ids_list = tokenizer.encode_graph(graph)
            src_ids = torch.tensor([src_ids_list], dtype=torch.long)
            raw_tokens = _greedy_decode(model, src_ids, tokenizer, device, 60)
            model_path = _decode_token_sequence(raw_tokens, tokenizer)
            if model_path is None: model_path = []
            
            res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)
            
            d = stats[b_key]
            d["n"] += 1
            
            # (a) share with path[0] == source
            if len(model_path) > 0 and model_path[0] == graph.source:
                d["path0_correct"] += 1
                
            # (b) EOS vs max_decode_len
            if tokenizer.eos_token_id in raw_tokens:
                d["eos_ended"] += 1
            else:
                d["max_ended"] += 1
                
            # (c) lengths
            d["gen_lengths"].append(len(model_path))
            d["dj_lengths"].append(len(sp.path))
            
            # (d) non-exclusive counts
            v_edges = res.edges_valid
            is_simple = len(model_path) == len(set(model_path))
            ends_tgt = (len(model_path) > 0 and model_path[-1] == graph.target)
            
            if v_edges: d["edges_valid"] += 1
            if is_simple: d["simple_path"] += 1
            if ends_tgt: d["ends_at_target"] += 1
            if v_edges and is_simple and ends_tgt: d["all_three"] += 1
            
            # (e) last node adjacent to target
            last_node = model_path[-1] if len(model_path) > 0 else None
            is_adj = False
            if last_node is not None and last_node != graph.target:
                adj = graph.adjacency()
                for v, _ in adj.get(last_node, []):
                    if v == graph.target:
                        is_adj = True
                        break
            if is_adj:
                d["last_adj_target"] += 1
                
            # Print 5 OOD examples
            if "OOD" in split_name and examples_printed < 5:
                print(f"Example {examples_printed+1}: N={graph.num_nodes}, Src={graph.source}, Tgt={graph.target}")
                tok_strs = [tokenizer.id_to_token.get(tid, str(tid)) for tid in raw_tokens]
                print(f"  Raw Tokens: {' '.join(tok_strs)}")
                print(f"  Model Path: {model_path}")
                print(f"  Dijkstra Path: {sp.path}")
                examples_printed += 1
                
        for b in bins:
            d = stats[b]
            count = d["n"]
            if count == 0: continue
            print(f"\n[BIN {b[0]}-{b[1]}] N={count}")
            print(f"  (a) path[0] == source: {d['path0_correct']} ({100.0*d['path0_correct']/count:.1f}%)")
            print(f"  (b) Ended with EOS: {d['eos_ended']} ({100.0*d['eos_ended']/count:.1f}%), Hit max_decode_len: {d['max_ended']} ({100.0*d['max_ended']/count:.1f}%)")
            g_len = d['gen_lengths']
            d_len = d['dj_lengths']
            print(f"  (c) Gen Length - Mean: {np.mean(g_len):.1f}, Median: {np.median(g_len):.1f}")
            print(f"      Djk Length - Mean: {np.mean(d_len):.1f}, Median: {np.median(d_len):.1f}")
            print(f"  (d) edges_valid: {d['edges_valid']} ({100.0*d['edges_valid']/count:.1f}%)")
            print(f"      simple_path: {d['simple_path']} ({100.0*d['simple_path']/count:.1f}%)")
            print(f"      ends_at_tgt: {d['ends_at_target']} ({100.0*d['ends_at_target']/count:.1f}%)")
            print(f"      Meets all 3: {d['all_three']} ({100.0*d['all_three']/count:.1f}%)")
            print(f"  (e) Last node adj to tgt: {d['last_adj_target']} ({100.0*d['last_adj_target']/count:.1f}%)")

    # ID
    print("Generating ID set...")
    run_eval("ID (5-20)", generate_dataset_split(1000, (5, 20), 1))
    
    # OOD
    print("\nGenerating OOD set...")
    run_eval("OOD (25-50)", generate_dataset_split(3000, (25, 50), 2))
    
    # Size sweep
    print("\n=======================================================")
    print(" SIZE SWEEP DIAGNOSTIC (N = 18 to 30)")
    print("=======================================================")
    sweep_ns = [18, 20, 21, 22, 23, 24, 25, 26, 28, 30]
    for n in sweep_ns:
        swp = generate_dataset_split(num_examples=200, node_range=(n, n), base_seed=3)
        vo = ce = ev = 0
        for graph, sp in swp.examples:
            res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)
            if res.valid_and_optimal: vo += 1
            if res.correct_endpoints: ce += 1
            if res.edges_valid: ev += 1
        print(f"N={n:2d} (200 graphs) | V&O: {vo:3d} ({100.0*vo/200:5.1f}%) | correct_ends: {ce:3d} ({100.0*ce/200:5.1f}%) | edges_valid: {ev:3d} ({100.0*ev/200:5.1f}%)")

if __name__ == "__main__":
    main()

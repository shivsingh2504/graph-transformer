import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import torch
from data.graph_generator import generate_random_connected_graph
from data.dijkstra import run_dijkstra
from data.tokenizer import GraphTokenizer
from eval.evaluate import evaluate_example
from model.model import Transformer

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

CKPT_PATH = os.path.join(_HERE, "..", "checkpoints_run5", "final.pt")
OUT_PATH = os.path.join(_HERE, "..", "results", "m9b_run5.txt")

def main():
    f_out = open(OUT_PATH, "w")
    sys.stdout = Tee(sys.stdout, f_out)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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

    def run_experiment(name, configs):
        print(f"\n=======================================================")
        print(f" {name}")
        print(f"=======================================================")
        
        last_group = None
        for group, n, num_edges, k, desc in configs:
            if group != last_group:
                print(f"\n--- {group} ---")
                last_group = group
                
            vo = ce = ev = vp = oc = 0
            num_examples = 400
            dj_hops = []
            
            tok_len = 3 * num_edges + 4
            density = num_edges / (n * (n - 1) / 2) if n > 1 else 0
            print(f"\n{desc} - {num_examples} graphs")
            print(f"  Token length: {tok_len}, Density: {density:.3f}")
            
            for i in range(num_examples):
                seed = 20_000_000 + 100_000 * k + i
                graph = generate_random_connected_graph(
                    num_nodes=n,
                    num_edges=num_edges,
                    seed=seed,
                    min_weight=tok_kwargs.get("min_weight", 1),
                    max_weight=tok_kwargs.get("max_weight", 10)
                )
                sp = run_dijkstra(graph)
                res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)
                
                if res.valid_and_optimal: vo += 1
                if res.correct_endpoints: ce += 1
                if res.edges_valid: ev += 1
                if res.valid_path: vp += 1
                if res.optimal_cost: oc += 1
                dj_hops.append(len(sp.path) - 1)
                
            mean_dj = sum(dj_hops) / len(dj_hops) if dj_hops else 0.0
            print(f"  Metrics:")
            print(f"    valid_and_optimal : {vo:4d} ({100.0*vo/num_examples:>5.1f}%)")
            print(f"    correct_endpoints : {ce:4d} ({100.0*ce/num_examples:>5.1f}%)")
            print(f"    edges_valid       : {ev:4d} ({100.0*ev/num_examples:>5.1f}%)")
            print(f"    valid_path        : {vp:4d} ({100.0*vp/num_examples:>5.1f}%)")
            print(f"    optimal_cost      : {oc:4d} ({100.0*oc/num_examples:>5.1f}%)")
            print(f"  Mean Dijkstra hops  : {mean_dj:.2f}")

    configs = [
        # Size generalization at trained input length (N = 25..50, E=60)
        ("Size Generalization (Fixed length 184)", 25, 60, 0, "Size Gen: N=25, E=60 (len=184)"),
        ("Size Generalization (Fixed length 184)", 30, 60, 1, "Size Gen: N=30, E=60 (len=184)"),
        ("Size Generalization (Fixed length 184)", 35, 60, 2, "Size Gen: N=35, E=60 (len=184)"),
        ("Size Generalization (Fixed length 184)", 40, 60, 3, "Size Gen: N=40, E=60 (len=184)"),
        ("Size Generalization (Fixed length 184)", 45, 60, 4, "Size Gen: N=45, E=60 (len=184)"),
        ("Size Generalization (Fixed length 184)", 50, 60, 5, "Size Gen: N=50, E=60 (len=184)"),
        
        # Discriminating test 1: unseen multiples of 3 vs non-multiples
        ("Discriminating test 1 (Unseen multiples vs non-multiples of 3 at N=7)", 7, 12, 6, "Unseen Multiple: N=7, E=12 (len=40)"),
        ("Discriminating test 1 (Unseen multiples vs non-multiples of 3 at N=7)", 7, 13, 17, "Contrast: N=7, E=13 (len=43)"),
        ("Discriminating test 1 (Unseen multiples vs non-multiples of 3 at N=7)", 7, 18, 7, "Unseen Multiple: N=7, E=18 (len=58)"),
        ("Discriminating test 1 (Unseen multiples vs non-multiples of 3 at N=7)", 7, 19, 18, "Contrast: N=7, E=19 (len=61)"),
        
        # Discriminating test 2: unseen multiples > 60 at n=20
        # Note: These repeat the earlier N=22-24 sweep, which collapsed (4.0%, 1.5%, 0%)
        ("Discriminating test 2 (Unseen multiples E>60 at N=20 - repeats earlier N=22-24 collapse)", 20, 66, 8, "Unseen Multiple: N=20, E=66 (len=202)"),
        ("Discriminating test 2 (Unseen multiples E>60 at N=20 - repeats earlier N=22-24 collapse)", 20, 69, 9, "Unseen Multiple: N=20, E=69 (len=211)"),
        ("Discriminating test 2 (Unseen multiples E>60 at N=20 - repeats earlier N=22-24 collapse)", 20, 72, 10, "Unseen Multiple: N=20, E=72 (len=220)"),
        
        # Discriminating test 3: one sparse n=20 row at each trained E from 45 to 60
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 45, 11, "Trained Length: N=20, E=45 (len=139)"),
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 48, 12, "Trained Length: N=20, E=48 (len=148)"),
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 51, 13, "Trained Length: N=20, E=51 (len=157)"),
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 54, 14, "Trained Length: N=20, E=54 (len=166)"),
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 57, 15, "Trained Length: N=20, E=57 (len=175)"),
        ("Discriminating test 3 (Sparse N=20 rows at trained lengths)", 20, 60, 16, "Trained Length: N=20, E=60 (len=184)"),
    ]
    
    run_experiment("M9b DIAGNOSTIC", configs)

if __name__ == "__main__":
    main()

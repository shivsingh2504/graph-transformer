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

CKPT_PATH = os.path.join(_HERE, "..", "checkpoints_run5", "final.pt")

def main():
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
        
        for n, num_edges, k, desc in configs:
            vo = ce = ev = 0
            num_examples = 200
            dj_lengths = []
            
            tok_len = 3 * num_edges + 4
            print(f"\n{desc} - 200 graphs")
            
            for i in range(num_examples):
                seed = 7_000_000 + 100_000 * k + i
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
                dj_lengths.append(len(sp.path))
                
            mean_dj = sum(dj_lengths) / len(dj_lengths) if dj_lengths else 0.0
            print(f"  V&O: {vo:3d} ({100.0*vo/200:5.1f}%) | correct_ends: {ce:3d} ({100.0*ce/200:5.1f}%) | edges_valid: {ev:3d} ({100.0*ev/200:5.1f}%)")
            print(f"  Mean Dijkstra path length: {mean_dj:.1f}")

    configs = [
        # BASELINES (ID sizes, ID lengths E=3n)
        (15, 45, 0, "Baseline (ID N=15, E=45, len=139)"),
        (20, 60, 1, "Baseline (ID N=20, E=60, len=184)"),
        
        # SPARSE CONTROLS (ID sizes, sparse lengths)
        (15, 30, 2, "Control Sparse (ID N=15, E=30, len=94)"),
        (20, 40, 3, "Control Sparse (ID N=20, E=40, len=124)"),
        (20, 40, 4, "Control Sparse Repeat (ID N=20, E=40, len=124)"),
        
        # DENSE CONTROL (ID sizes, dense length)
        (12, 60, 5, "Control Dense (ID N=12, E=60, len=184)"),
        
        # COMB TEST (N=20, E=54 to 63)
        (20, 54, 6, "Comb Test (N=20, E=54, len=166)"),
        (20, 55, 7, "Comb Test (N=20, E=55, len=169)"),
        (20, 56, 8, "Comb Test (N=20, E=56, len=172)"),
        (20, 57, 9, "Comb Test (N=20, E=57, len=175)"),
        (20, 58, 10, "Comb Test (N=20, E=58, len=178)"),
        (20, 59, 11, "Comb Test (N=20, E=59, len=181)"),
        (20, 60, 12, "Comb Test (N=20, E=60, len=184)"),
        (20, 61, 13, "Comb Test (N=20, E=61, len=187)"),
        (20, 62, 14, "Comb Test (N=20, E=62, len=190)"),
        (20, 63, 15, "Comb Test (N=20, E=63, len=193)"),
        
        # SIZE TEST (N=25 to 50 at E=60)
        (25, 60, 16, "Size Test (N=25, E=60, len=184)"),
        (30, 60, 17, "Size Test (N=30, E=60, len=184)"),
        (35, 60, 18, "Size Test (N=35, E=60, len=184)"),
        (40, 60, 19, "Size Test (N=40, E=60, len=184)"),
        (45, 60, 20, "Size Test (N=45, E=60, len=184)"),
        (50, 60, 21, "Size Test (N=50, E=60, len=184)"),
        
        # SET A (ID nodes, OOD length)
        (15, 75, 22, "SET A: ID Nodes (N=15), OOD Length (E=75, len=229)"),
        (20, 100, 23, "SET A: ID Nodes (N=20), OOD Length (E=100, len=304)"),
        
        # SET B (OOD nodes, ID length)
        (22, 40, 24, "SET B: OOD Nodes (N=22), ID Length (E=40, len=124)"),
        (25, 50, 25, "SET B: OOD Nodes (N=25), ID Length (E=50, len=154)"),
        (30, 60, 26, "SET B: OOD Nodes (N=30), ID Length (E=60, len=184)"),
    ]
    
    run_experiment("DIAGNOSTIC SWEEPS", configs)

if __name__ == "__main__":
    main()

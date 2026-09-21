"""Extra N=50, E=60 (len=184) block on a fresh seed range.

Settles the disagreement between the two existing N=50 measurements:
  diagnostics/diagnostic_length_vs_nodes.py  k=21 -> seeds  9,100,000+i, n=200 -> 15.0%
  diagnostics/m9b_eval.py                     k=5  -> seeds 20,500,000+i, n=400 -> 29.2%

Uses the identical generator and evaluation calls as m9b_eval.py so the three
blocks are directly comparable. Evaluation only; trains nothing.
"""
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
OUT_PATH = os.path.join(_HERE, "..", "results", "n50_extra_block.txt")

NUM_NODES = 50
NUM_EDGES = 60
NUM_EXAMPLES = 1000
SEED_BASE = 30_000_000


def main():
    f_out = open(OUT_PATH, "w")

    def emit(line=""):
        print(line)
        f_out.write(line + "\n")
        f_out.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit(f"Device: {device}")

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

    tok_len = 3 * NUM_EDGES + 4
    density = NUM_EDGES / (NUM_NODES * (NUM_NODES - 1) / 2)
    emit(f"\nN={NUM_NODES}, E={NUM_EDGES} (len={tok_len}) - {NUM_EXAMPLES} graphs")
    emit(f"  Seed range: {SEED_BASE} .. {SEED_BASE + NUM_EXAMPLES - 1}")
    emit(f"  Token length: {tok_len}, Density: {density:.3f}")

    vo = ce = ev = vp = oc = 0
    half_vo = [0, 0]
    dj_hops = []

    for i in range(NUM_EXAMPLES):
        seed = SEED_BASE + i
        graph = generate_random_connected_graph(
            num_nodes=NUM_NODES,
            num_edges=NUM_EDGES,
            seed=seed,
            min_weight=tok_kwargs.get("min_weight", 1),
            max_weight=tok_kwargs.get("max_weight", 10),
        )
        sp = run_dijkstra(graph)
        res = evaluate_example(model, graph, sp, tokenizer, device, max_decode_len=60)

        vo += int(res.valid_and_optimal)
        half_vo[0 if i < NUM_EXAMPLES // 2 else 1] += int(res.valid_and_optimal)
        ce += int(res.correct_endpoints)
        ev += int(res.edges_valid)
        vp += int(res.valid_path)
        oc += int(res.optimal_cost)
        dj_hops.append(len(sp.path) - 1)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{NUM_EXAMPLES} ...", flush=True)

    n = NUM_EXAMPLES
    mean_dj = sum(dj_hops) / len(dj_hops)
    emit("  Metrics:")
    emit(f"    valid_and_optimal : {vo:4d} ({100.0*vo/n:>5.1f}%)")
    emit(f"    correct_endpoints : {ce:4d} ({100.0*ce/n:>5.1f}%)")
    emit(f"    edges_valid       : {ev:4d} ({100.0*ev/n:>5.1f}%)")
    emit(f"    valid_path        : {vp:4d} ({100.0*vp/n:>5.1f}%)")
    emit(f"    optimal_cost      : {oc:4d} ({100.0*oc/n:>5.1f}%)")
    emit(f"  Mean Dijkstra hops  : {mean_dj:.2f}")

    # Within-block stability: the two existing N=50 measurements disagree by
    # ~4 sigma, so check whether a single large block is itself homogeneous.
    h = NUM_EXAMPLES // 2
    emit("\n  Half-split (within-block stability):")
    for label, count in (("first ", half_vo[0]), ("second", half_vo[1])):
        emit(f"    {label} {h}: {count:4d} ({100.0*count/h:>5.1f}%)")

    f_out.close()


if __name__ == "__main__":
    main()

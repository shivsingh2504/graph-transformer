"""Reproduce the two data-integrity hashes quoted in run_log.md.

Amendment 2 pins run 6's training data to a graph-level SHA-256; the run 7 lock
ruling additionally pins the tokenisation of the same 200 graphs. Until now both
values existed only as text, so this script is the reproducible recipe. It exits
non-zero on mismatch so it can gate a training launch.

    python diagnostics/verify_data_hashes.py
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer

PROBE_EXAMPLES = 200
PROBE_NODE_RANGE = (5, 20)
PROBE_BASE_SEED = 0

EXPECTED_GRAPH = (
    "bf1adf1930c5a1d8d542b91319fca6b261c7109e8fb740fc55b73c1683fd9278"
)
EXPECTED_TOKENS = (
    "0d7c7fdbfad17308703084ec400687f1e85426c5df07c3f547de9f1a256d0c88"
)


def graph_hash(split) -> str:
    payload = json.dumps(
        [
            [g.num_nodes, g.node_ids, [list(e) for e in g.edges],
             g.source, g.target, g.seed, sp.cost, sp.path]
            for g, sp in split.examples
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def token_hash(split, tokenizer: GraphTokenizer) -> tuple[str, int, int]:
    ids = [tokenizer.encode_graph(g) for g, _ in split.examples]
    strs = [[tokenizer.id_to_token[i] for i in seq] for seq in ids]
    payload = json.dumps({"ids": ids, "strs": strs}, sort_keys=True)
    lengths = [len(seq) for seq in ids]
    return (
        hashlib.sha256(payload.encode()).hexdigest(),
        min(lengths),
        max(lengths),
    )


def main() -> int:
    split = generate_dataset_split(
        num_examples=PROBE_EXAMPLES,
        node_range=PROBE_NODE_RANGE,
        base_seed=PROBE_BASE_SEED,
    )
    tokenizer = GraphTokenizer(min_weight=1, max_weight=10)

    got_graph = graph_hash(split)
    got_tokens, min_len, max_len = token_hash(split, tokenizer)

    print(f"probe: first {PROBE_EXAMPLES} train graphs, "
          f"node_range={PROBE_NODE_RANGE}, base_seed={PROBE_BASE_SEED}")
    print(f"vocab_size={tokenizer.vocab_size} "
          f"encoder length min/max={min_len}/{max_len}")
    print()
    ok = True
    for label, got, want in (
        ("graph data  ", got_graph, EXPECTED_GRAPH),
        ("tokenisation", got_tokens, EXPECTED_TOKENS),
    ):
        match = got == want
        ok = ok and match
        print(f"{label} {'MATCH  ' if match else 'MISMATCH'} {got}")
        if not match:
            print(f"{'':12s}expected {want}")

    print()
    if ok:
        print("MATCH on data AND tokenization")
        return 0
    print("Data does not match the locked spec. Do not train.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

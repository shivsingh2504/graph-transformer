# Project Run Log

This file records the run history, key definitions, and evaluation notes for the graph-transformer shortest-path project.

## Run history

| Run | Checkpoint | Train Data | Epochs | Val loss | V&O |
|---|---|---|---|---|---|
| Run 1 | `checkpoints_run1/final.pt` (tree baseline) | - | - | - | - |
| Run 2 | `checkpoints_run2/final.pt` (tree baseline) | - | - | - | 92.7% |
| Run 3 | `checkpoints_run3/final.pt` | 50,000 | 30 | 1.51 | 24.8% |
| Run 4 | `checkpoints_run4/final.pt` (commit `9dbd4af`) | 500,000 | 10 | 0.59 | 46.3% |
| Run 5 | `checkpoints_run5/final.pt` (commit `fb37f42`) | 500,000 | 30 | 0.129 | 83.6% |

## Metric definitions

### weight_sensitive

A graph is **weight-sensitive** if the hop-count of the Dijkstra-optimal path
differs from the BFS minimum hop count:

```
weight_sensitive(graph, sp) = (len(sp.path) - 1) != bfs_min_hops(graph.source, graph.target)
```

where `sp` is the result of `run_dijkstra(graph)` and `bfs_min_hops` is an
unweighted BFS returning the minimum number of edges on any source-to-target path.

**Implementation:** `_weight_sensitive_datacheck()` in `breakdown_run5.py`.
This definition matches `data_check.py` (line 80): `min_cost_hops != min_hop`.

**Caveat:** `run_dijkstra` returns one optimal path when ties exist; integer
weights in [1, 10] make ties common. A graph where two equal-cost paths have
different hop counts will be classified as sensitive based on whichever path
Dijkstra returns.

**Reference numbers:**
- Train split (50,000 graphs, seed 0): **36.7%** sensitive
- ID eval split (1,000 graphs, seed 1): **36.8%** sensitive

## Amendments & Waivers

### Amendment 1: gate waiver, 2026-09-21
Run 5 (commit `fb37f42`, 500,000 train examples, 30 epochs, otherwise the run 3 spec) scored 83.6% valid_and_optimal on the ID eval, below the >=90% gate. The gate is waived so M9 OOD evaluation can proceed on `checkpoints_run5\final.pt`. Every OOD report must state the 83.6% ID score and ID accuracy by size (95.1% / 82.5% / 75.7%). All other M9 terms unchanged.
**Authorized by:** project owner, who delegated the decision to the reviewer on 2026-09-21.

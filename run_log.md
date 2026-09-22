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

### Amendment 2: run 6 epoch count, 2026-09-22
Run 6 changes exactly one variable from the run 5 spec (`fb37f42`): `n_epochs`
30 -> 60. Every other `RunConfig` field is inherited unchanged - 500,000 train
examples, 1,000 eval examples, `node_range` (5, 20), train_seed 0, eval_seed 1,
batch_size 32, lr 5e-4, weight_decay 0.01, warmup_steps 4,000, grad_clip_norm
1.0, 3 layers, d_model 128, 4 heads, d_ff 512, dropout 0.0.
`src/run_training_run6.py` asserts at startup that `n_epochs` is the only field
that differs, and writes to `checkpoints_run6/`.

**Why this variable.** Run 5 was undertrained, not overfit: final train loss
0.1224 against final val loss 0.1285, val loss still falling across the last 10
epochs (0.1481 -> 0.1285), and the best val loss of the run was the last epoch.
Runs 4 and 5 hold the locked distribution and the 500k train split fixed and
differ only in epoch count, and 10 -> 30 epochs moved valid_and_optimal from
46.3% to 83.6%, so epoch count is the one lever with direct evidence on this
distribution.

**The LR schedule does not reshape.** `_linear_warmup_schedule` (train/train.py)
ramps linearly to 1.0 over `warmup_steps` and returns a constant 1.0 afterwards;
it takes no `n_epochs` or total-step argument, so doubling the epoch count does
not alter the schedule's shape. This is what keeps the run single-variable.

**Hardware.** Run 5 was trained on Kaggle (3.27 h wall time for 30 epochs), so
run 6 is expected to take roughly 6.5 h there, inside the 12 h session limit.
Training data is generated from `random.Random(base_seed * 1_000_000 + i)`
(CPython Mersenne Twister), which is platform-independent, so the 500k examples
are identical wherever the run executes; the Kaggle notebook verifies this by
hashing the first 200 training graphs and comparing against the locally computed
`bf1adf1930c5a1d8d542b91319fca6b261c7109e8fb740fc55b73c1683fd9278`. Floating
point arithmetic on an accelerator is not bit-identical to CPU, so run 6 is not a
bit-exact continuation of run 5's optimisation trajectory. To keep reported
figures comparable with run 5's, every run 6 evaluation number will be recomputed
locally on CPU from the downloaded `checkpoints_run6/final.pt` using the existing
`diagnostics/m9b_eval.py` and `src/run_ood_eval.py`, rather than read off the
training host.

**Gate.** Amendment 1 waived the >=90% gate for run 5 only. Run 6 is subject to
the gate again: it passes if ID valid_and_optimal >= 90%.

**Authorized by:** project owner on 2026-09-22, who delegated the choice of
variable ("ur call") to the reviewer.

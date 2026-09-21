# Results

Graph-transformer shortest-path project. Status as of 2026-09-22, checkpoint
`checkpoints_run5/final.pt`.

**Headline.** Run 5 reaches **83.6%** valid-and-optimal in distribution, below the
90% gate (waived, see [Deviations](#deviations-from-the-lock)). Out of
distribution at N=25–50 it scores **0.0%** (0/3000). Diagnostics show the OOD
collapse is driven by **input token length**, not graph size: holding the input
at a trained length of 184 tokens, the model still scores 24–63% out to N=50,
while changing the length by three tokens at N=20 — a size it handles well —
drops it to 0%.

---

## Setup

**Task.** Given a weighted, undirected, connected graph plus a source and target,
emit the Dijkstra-optimal path as a sequence of node ids.

**Data.** `generate_random_connected_graph` builds a random spanning tree, adds
extra edges up to a target count, assigns integer weights in [1, 10], then
**relabels nodes to a random subset of `0..49`** — node ids are not contiguous and
carry no ordering information. Under the locked edge rule
(`src/data/dataset_generator.py:69`):

```
E = min(3N, N(N-1)/2)
```

Train and ID eval use `node_range=(5, 20)`; train seed 0 (500,000 examples), eval
seed 1 (1,000 examples). Cross-split fingerprint collisions: 0.

**Encoding.** `GraphTokenizer.encode_graph` emits canonically sorted `(u, v, w)`
triples, then appends the endpoints at the **end** of the sequence:

```
node_u node_v weight_w  ...  [SRC] node_s [DST] node_t
```

Input length is therefore exactly `3E + 4`. Vocab is 65 tokens (5 special, 50
node, 10 weight). Target is `[BOS] <path nodes> [EOS]`.

**Consequence of the locked edge rule.** Because `E` is a deterministic function
of `N`, every training example for a given `N` has exactly one input length. The
entire 500k-example training set contains only **16 distinct input lengths**:

| N | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | 10 | 15 | 21 | 24 | 27 | 30 | 33 | 36 | 39 | 42 | 45 | 48 | 51 | 54 | 57 | 60 |
| len | 34 | 49 | 67 | 76 | 85 | 94 | 103 | 112 | 121 | 130 | 139 | 148 | 157 | 166 | 175 | 184 |

For N≥8 these form an arithmetic progression with step 9. Verified by
enumerating the eval split, not inferred.

**Model.** Encoder–decoder transformer: 3 layers, `d_model=128`, 4 heads,
`d_ff=512`, dropout 0.0. Positional encoding is **sinusoidal with `max_len=5000`**
(`src/model/layers.py`) — there is no architectural length cap, so nothing about
length 229+ is out of range. Decoding is greedy, `max_decode_len=60`.

**Training.** batch 32, lr 5e-4, weight decay 0.01, warmup 4,000 steps, grad clip
1.0, 30 epochs. Run 5 wall time 11,771s (3h16m) on CUDA, torch 2.10.0+cu128.
All evaluation below was run on CPU.

**Metrics.** `valid_and_optimal` (V&O) is the headline: path is simple, every hop
is a real edge, endpoints match source and target, and total cost equals the
Dijkstra cost.

> **Caution on `optimal_cost`.** In `src/eval/evaluate.py`, `_check_path` computes
> `correct_endpoints` and `optimal_cost` **independently**. `optimal_cost` is
> `(decoded_cost == true_cost)`, where `decoded_cost` just sums the weights along
> whatever node sequence was emitted — it never references `graph.source` or
> `graph.target`, and does not require the path to be simple. So `optimal_cost`
> can be, and in the collapsed rows is, **larger than V&O while endpoints are
> near zero**: at N=7/len 61 it is 12.5% against 0.0% V&O and 0.0% endpoints, and
> at N=20/E=66 it is 8.0% against 1.2% V&O. Those are walks between the *wrong*
> node pair whose total weight coincidentally equals the true optimum — easy to
> hit with integer weights in [1, 10] and true costs of only a few units.
> **`optimal_cost` is not partial credit and must not be read as evidence the
> model is nearly working in a collapsed row.** Only `correct_endpoints` and V&O
> are.

---

## Run history

| Run | Checkpoint | Data | Train ex. | Epochs | lr | Final val loss | ID V&O | Gate | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `checkpoints_run1/final.pt` | **trees** (E=N−1) | 50,000 | 30 | 1e-4 | 0.266 | 62.9% | fail | `results.json` (committed) |
| 2 | `checkpoints_run2/final.pt` | **trees** (E=N−1) | 50,000 | 30 | 5e-4 | 0.049 | **92.7%** | pass | `results.json` (committed) |
| 3 | `checkpoints_run3/final.pt` | locked | 50,000 | 30 | — | 1.51 | 24.8% | fail | `run_log.md` only |
| 4 | `checkpoints_run4/final.pt` | locked | 500,000 | 10 | — | 0.59 | 46.3% | fail | `run_log.md` only |
| 5 | `checkpoints_run5/final.pt` | locked | 500,000 | 30 | 5e-4 | 0.129 | **83.6%** | fail (waived) | `results.json` |

Runs 1–2 were confirmed at 50,000 examples / 30 epochs from their committed
`results.json` files. **They are not comparable to runs 3–5.** The locked edge
rule landed in commit `160eaaf` (2026-09-20 17:04), after run 2's checkpoint was
written (15:19). Before that commit `generate_dataset_split` passed no edge
count, and `generate_random_connected_graph` defaults to
`target_edge_count = min_spanning_edges = N-1` — a tree. `run_log.md` already
labels them "(tree baseline)".

This matters: on a tree the shortest path is unique and the task is
weight-insensitive, so **run 2's 92.7% gate pass does not count against the
locked distribution.** The 90% gate has never been met on the locked data. Run
2's ID eval split has the same per-N counts (304/297/399 across the 5–9 / 10–14 /
15–20 bins) as run 5's, so the node distribution is identical — only the edge
counts differ.

Runs 3 and 4 have no artifacts on disk: `checkpoints_run3/` is empty and
`checkpoints_run4/` does not exist. Their figures come solely from `run_log.md`
and cannot be re-verified.

---

## In-distribution results (run 5)

1,000 graphs, seed 1, N=5–20. Re-run and confirmed against
`checkpoints_run5/results.json`:

| Metric | Count | % |
|---|---|---|
| valid_and_optimal | 836 | **83.6** |
| correct_endpoints | 986 | 98.6 |
| edges_valid | 970 | 97.0 |
| valid_path | 952 | 95.2 |
| optimal_cost | 836 | 83.6 |

By size:

| Bin | N | V&O |
|---|---|---|
| 5–9 | 304 | **95.1%** (289/304) |
| 10–14 | 297 | **82.5%** (245/297) |
| 15–20 | 399 | **75.7%** (302/399) |

Failure taxonomy over the 164 failures (exclusive, checked in this order):
suboptimal 113, invalid edge 21, cycle 16, wrong endpoint 14. Structural failures
(wrong endpoint + invalid edge + cycle) are 51 of 164; cost errors are the other
113. Cross-check: 116 graphs have `valid_path=True` but are not optimal = 113
suboptimal + 3 that are valid paths with wrong endpoints.

Weight sensitivity (Def A, the locked definition — Dijkstra hop count differs from
BFS minimum hops): 368 of 1,000 ID graphs (36.8%) are weight-sensitive, matching
the 36.7% measured on the train split. Accuracy is **71.2%** on sensitive graphs
vs **90.8%** on insensitive ones, a 19.6-point gap. So even in distribution the
model is materially worse exactly where weights override hop count — the
capability the task is meant to test. (Def B, a non-locked BFS-cost comparison
definition, gives 50.5% sensitive and a 16.5-point gap; reported in
`results/breakdown_run5.txt` for reference only.)

In distribution the model almost always identifies the right endpoints (98.6%).
The dominant failure is a real, simple path between the correct endpoints that is
merely **not cost-optimal** — 113 of 164 failures. Endpoint binding is not the
problem here; it becomes the entire problem out of distribution.

---

## Out-of-distribution results (M9)

`src/run_ood_eval.py`: 3,000 graphs, `node_range=(25, 50)`, seed 2. The script
first re-runs the ID eval and asserts 836/1000 before proceeding (control step
passed).

Because the same locked edge rule applies, the OOD set has E = 3N, giving input
lengths **229 to 454 tokens** — entirely above the trained maximum of 184.

| Bin | n | V&O | correct_endpoints | edges_valid | valid_path | optimal_cost |
|---|---|---|---|---|---|---|
| 25–29 | 613 | 0.0% | 0.0% | 79.4% | 37.0% | 5.9% |
| 30–34 | 586 | 0.0% | 0.0% | 61.4% | 24.7% | 2.4% |
| 35–39 | 572 | 0.0% | 0.3% | 54.7% | 20.5% | 2.8% |
| 40–44 | 579 | 0.0% | 0.5% | 45.1% | 14.0% | 2.6% |
| 45–50 | 650 | 0.0% | 0.0% | 46.2% | 17.5% | 2.5% |
| **Overall** | **3000** | **0.0%** | **0.2%** | **57.4%** | **22.8%** | **3.2%** |

Failure taxonomy: **99.8% wrong endpoint** (2,995 of 3,000), 4 invalid edge, 1
cycle, 0 suboptimal. Weight-sensitive graphs are 42.8% of the OOD set; accuracy
on those is 0/1283.

The `optimal_cost` column above (2.4–5.9% per bin, 3.2% overall) is **not**
partial success — see the caution in [Setup](#setup). With endpoints correct in
only 0.2% of cases, those are coincidental cost matches on walks between the
wrong node pair.

Decoded behaviour (`results/ood_diagnostics_run5.txt`): 98–99% of generations
terminate with EOS rather than hitting the decode cap, so the model is confident
and fluent — it just answers a different question. `path[0] == source` holds in
only 3.6–13.8% of cases. Generated paths are longer than Dijkstra's (mean 5.8–7.9
vs 3.5–4.0) and frequently revisit nodes:

```
Example 1: N=42, Src=46, Tgt=39
  Model Path:    [42, 38, 2, 22, 2, 49, 2, 26]
  Dijkstra Path: [46, 35, 39]
```

Note `edges_valid` stays at 45–79%: the model has largely parsed the edge list
correctly and emits real edges. What it cannot do is find the source and target.

---

## What actually drives the collapse: length, not size

The OOD test above changes node count and input length together, so it cannot
attribute the failure. Two diagnostics separate them.

**Vary size at a trained length** (`results/m9b_run5.txt`, E=60, len=184, n=400
per row):

| N | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|
| V&O | 63.0% | 47.8% | 39.8% | 31.8% | 26.2% | 29.2% |
| correct_endpoints | 95.8% | 89.8% | 85.8% | 83.5% | 84.0% | 79.5% |
| edges_valid | 80.8% | 63.5% | 51.5% | 41.2% | 34.0% | 35.2% |
| density | 0.200 | 0.138 | 0.101 | 0.077 | 0.061 | 0.049 |

Size generalises **gracefully**. At 2.5× the trained node count the model still
gets endpoints right 80–96% of the time. Degradation tracks density: at fixed
E=60 a larger graph is sparser, paths get longer (mean Dijkstra hops 2.66 → 4.34),
and the model increasingly invents edges. There is no cliff.

The N=45 → N=50 step (26.2% → 29.2%) is **not a rebound**. At n=400 the per-row
SE is ≈2.2 points, so those two rows are within 1σ of each other; read the sweep
as monotone-decreasing-plus-noise. The overall trend is real (63.0% → ~28% across
the sweep is many σ), individual adjacent pairs are not.

The N=50 cell is the least reliable entry in this table. A dedicated 1,000-graph
block on a fresh seed range gives **23.0%**, and pooling all three N=50
measurements (1,600 graphs) gives **23.6%** — see Limitation 5. The 29.2% shown
here is quoted from `m9b_run5.txt` for consistency with that file.

**Vary length at a trained size** (N=20, n=200–400 per row):

| E | 40 | 45 | 48 | 51 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 66 | 69 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| len | 124 | 139 | 148 | 157 | 166 | 169 | 172 | 175 | 178 | 181 | 184 | 187 | 190 | 193 | 202 | 211 | 220 |
| trained len? | no | yes | yes | yes | yes | no | no | yes | no | no | yes | no | no | no | no | no | no |
| V&O | **0%** | 65.0% | 60.0% | 70.2% | 69.0% | **0%** | **0%** | 68.5% | **0%** | **0%** | 71.8% | **0%** | 2.0% | 35.5% | 1.2% | 0% | 0.2% |
| correct_ends | 0% | 94.2% | 95.8% | 96.2% | 95.5% | 0% | 0.5% | 95.8% | 0% | 0% | 98.5% | 0% | 4.5% | 55.0% | 3.0% | 0.2% | 0.2% |
| source | comb | m9b | m9b | m9b | m9b | comb | comb | m9b | comb | comb | m9b | comb | comb | comb | m9b | m9b | m9b |
| n | 200 | 400 | 400 | 400 | 400 | 200 | 200 | 400 | 200 | 200 | 400 | 200 | 200 | 200 | 400 | 400 | 400 |

`comb` = `results/comb_and_size_run5.txt` (seeds 7,000,000+), `m9b` =
`results/m9b_run5.txt` (seeds 20,000,000+). **The table mixes two sample sizes
and two disjoint seed ranges**, so cross-row comparisons at the 1–3 point level
are not meaningful; the 0%-vs-60%+ contrast is.

Three configurations were measured by both scripts, which bounds that
disagreement: E=54 → 69.0% vs 69.0%, E=57 → 73.0% vs 68.5%, E=60 → 78.0% vs
71.8%. The largest gap is 6.2 points at E=60, about 1.6σ — consistent with
sampling noise. Where they overlap the two scripts agree; the m9b (n=400) value
is quoted above as the better-powered one.

Every trained length in [139, 184] scores 60–72%. Off-grid lengths collapse to
~0%, and the collapse is specifically in `correct_endpoints` — 0–4.5% on every
off-grid row bar one — while `edges_valid` stays high (62.0–99.5%). A
three-token change, one extra edge, is the difference between 71.8% and 0%.

The E=40 cell is measured twice in `comb_and_size_run5.txt` ("Control Sparse" and
"Control Sparse Repeat", 200 graphs each). Both give 0.0% V&O and 0.0% endpoints;
`edges_valid` differs (71.5% vs 66.5%). The table quotes n=200 for one block.

The single exception to the endpoint collapse is E=63 (len 193), at 35.5% V&O and
55.0% endpoints — the one off-grid row that partially works. Limitation 4 records
further counter-examples at N=7.

**What the E=45…60 rows do and do not show.** They are a *control*, not a
discriminating test. All six edge counts are multiples of 3 and all six lengths
are in the trained set, so the two candidate explanations — "E must be a multiple
of 3" and "length must be a trained length" — coincide perfectly here and cannot
be told apart. What the rows do establish is that **low density is not the
cause**: E=45 at N=20 is density 0.237, sparser than the trained N=20 density of
0.316, and still scores 65.0%. They also show the model tolerates an (N, E)
pairing it never saw — in training N=20 always came with E=60 — as long as the
length is a trained one.

Do not rank these six rows against each other. E=60 (71.8%) vs E=48 (60.0%) is an
11.8-point gap, about 3.5σ at n=400, and E=60 is the one length where N=20 was
actually trained, so that row being best is expected. The other five are within
about 2σ of each other and of baseline; the ordering among them is noise.

The real discriminator is the interleaved off-grid rows. E=55, 56, 58, 59 give
lengths 169, 172, 178, 181 — all *inside* the trained length range [34, 184], all
untrained, and all at 0%. Just above the range, E=61 (len 187) is also 0% and
E=62 (len 190) is 2.0%. Note also E=63 (len 193, a multiple of 3 but not a
trained length) at 35.5%, against E=66/69/72 (also multiples of 3, lengths
202/211/220) at ≤1.2% — which argues against "multiple of 3" as the operative
rule and for the specific trained-length set, with a separate extrapolation
failure above 184.

The same signature appears at N=15 with E=75 (len 229), measured by both
diagnostic scripts on disjoint seed ranges: 0.5% V&O / 95.0% edges_valid
(`length_vs_nodes_run5.txt`) and 1.5% V&O / 96.0% edges_valid
(`comb_and_size_run5.txt`). Both are n=200, and both contrast sharply with the
N=15 baseline at the trained len 139 (75.0% V&O). It also appears in the N=18→30
sweep (`results/ood_diagnostics_run5.txt`, n=200 per row), where N=18 scores
74.5% and N=20 77.0% at trained lengths, N=21 (len 193) drops to 41.5%, and
N=22+ (len 202+) fall to ≤3%. That sweep looked like a size cliff; it is a length
cliff that happens to coincide with size because E = 3N.

### Mechanism (hypothesis)

The `[SRC]` and `[DST]` markers sit at positions `3E` and `3E+2` — the very end of
the input, at an offset that varies with E. With only 16 distinct training
lengths, spaced 9 apart above N=7, the model has never seen the endpoint markers
at an untrained offset. The simplest explanation of the data is that it learned a
positional cue tied to the trained length grid rather than a content-based rule
("attend to the unique `[SRC]` token, read the next token"), and that cue fails
off-grid. Sinusoidal PE with `max_len=5000` rules out an architectural cause;
this is a data-coverage failure.

The prediction that follows — and the reason run 6 would move `[SRC]`/`[DST]` to
the front of the input — is that a fixed marker offset should remove the
length sensitivity entirely. **This is untested.**

---

## Limitations

1. **The 90% gate was not met** (83.6%) and was waived. Everything OOD rests on a
   checkpoint that did not pass its own acceptance criterion.
2. **The M9 OOD test confounds graph size with input length.** E = 3N means
   N=25–50 is also length 229–454. The 0.0% figure cannot be attributed to size
   on its own, and should not be read as "the model cannot handle 50 nodes" — the
   fixed-length sweep shows it can, at 23.6% pooled (Limitation 5).
3. **Only 16 distinct training lengths.** Length robustness was never trained
   for, so its absence is unsurprising and is a property of the locked data spec
   as much as of the model.
4. **Counter-evidence to the length hypothesis.** At N=7, off-grid lengths do
   *not* collapse: len 40 → 60.2%, len 43 → 78.2%, len 58 → 93.0%, but len 61 →
   0.0%. Length 43 and length 61 are equally off-grid and differ by 78 points.
   At N=20, len 190 → 2.0% but len 193 → 35.5%. "Off-grid length" is clearly not
   sufficient on its own, and the mechanism above does not explain these rows.
5. **N=50 measurements disagree beyond sampling noise — resolved as real
   heterogeneity.** The same configuration (N=50, E=60, len=184) has now been
   measured three times on disjoint seed ranges:

   | Block | Seeds | V&O | SE |
   |---|---|---|---|
   | `diagnostic_length_vs_nodes.py` (k=21) | 9,100,000+ | 15.0% (30/200) | ±2.5 |
   | `m9b_eval.py` (k=5) | 20,500,000+ | 29.2% (117/400) | ±2.3 |
   | `n50_extra_block.py` | 30,000,000+ | 23.0% (230/1000) | ±1.3 |

   **Pooled estimate: 23.6% (377/1600).** Pairwise the blocks differ by 4.2σ,
   2.8σ and 2.4σ, and a heterogeneity test gives χ² = 15.50 on 2 df,
   **p = 0.0004** — so this is genuine between-block variation, not harness noise.
   Greedy decoding on CPU is deterministic and all three blocks were re-run and
   reproduced exactly, so the variation is in *which graphs were drawn*, not in
   the measurement. Within the 1,000-graph block the two halves agree closely
   (23.6% vs 22.4%, 0.5σ), confirming the instability is between blocks rather
   than drift within one.

   Cause unknown. At density 0.049 the specific 60 edges and the source/target
   pair vary a great deal between draws, and `edges_valid` at N=50 is only
   29.8–35.2%, so the score sits on a knife-edge of which topologies appear.
   **Practical consequence: treat the N=50 figure as 23.6% with a block-to-block
   spread of roughly 15–29%, and do not quote a single block.** The other size
   rows are consistent between the two scripts that measured them (`comb` at
   n=200, `m9b` at n=400): N=25 63.5 vs 63.0, N=30 47.5 vs 47.8, N=35 38.0 vs
   39.8, N=40 29.0 vs 31.8, N=45 22.5 vs 26.2. The largest of those gaps is 3.7
   points at N=45, which is 1.0σ on the combined standard error — noise. N=50 is
   14.2 points, 4.2σ, and is the outlier. Note the size sweep in the
   [length-vs-size section](#what-actually-drives-the-collapse-length-not-size)
   still reports the m9b row (29.2%) for consistency with that file; 23.0% is the
   better-powered number.
6. **Runs 3 and 4 are unverifiable.** No `results.json`, no checkpoints.
7. **Weight-sensitivity metric is tie-dependent.** `run_dijkstra` returns one
   optimal path when costs tie, and integer weights in [1, 10] make ties common,
   so a graph with two equal-cost paths of different hop counts is classified by
   whichever path Dijkstra happens to return. Documented in `run_log.md`.
8. **Environment drift.** Run 1 used torch 2.13.0+cpu, run 2 torch 2.11.0+cpu,
   run 5 torch 2.10.0+cu128. `requirements.txt` pins `torch==2.13.0`. Run 5 was
   not produced in the pinned environment.
9. **Checkpoints are not in the repo.** `*.pt` and `checkpoints_*/` are
   gitignored, so no run can be re-evaluated from a clone — only retrained. Every
   evaluation number above depends on `checkpoints_run5/final.pt` (10.8 MB), which
   exists **only on the local machine** at
   `C:\Users\reals.DESKTOP-B4CI38M\Desktop\graph\checkpoints_run5\final.pt` and
   is not backed up anywhere in version control. The 2 KB
   `checkpoints_run5/results.json` *is* committed, so run 5's config and headline
   metrics are auditable even though the weights are not. Runs 1 and 2 likewise
   have committed `results.json` / `diagnostic.json` but their `.pt` files are
   local-only. Retraining run 5 on different hardware or a different torch build
   will not reproduce these numbers bit-for-bit.

---

## Deviations from the lock

`lock.md` is referenced by `diagnostics/breakdown_run5.py` but **is not present in
the repository**, so deviations cannot be checked against the authoritative lock
text. The following are recorded against `run_log.md` and the committed configs:

- **Gate waiver (Amendment 1, 2026-09-21).** Run 5 scored 83.6% ID V&O against a
  ≥90% gate. The gate was waived so M9 OOD evaluation could proceed on
  `checkpoints_run5/final.pt`. Authorized by the project owner, who delegated the
  decision to the reviewer. The waiver requires every OOD report to state the
  83.6% ID score and ID accuracy by size (95.1% / 82.5% / 75.7%); those figures
  appear above.
- **Runs 1–2 are pre-lock.** They trained on trees, not the locked edge rule, and
  are reported here as baseline context only.
- **Torch version deviates from the pin** for runs 2 and 5 (see Limitation 8).
- **`src/api.py` retains a `TEST_RUN2_MODE` flag** with a separate hand-rolled
  tree generator for the run-2 code path. It is set to `False`, so the demo serves
  run 5, but the dead branch is still in the shipped API.

---

## Reproducing

Windows, from the repo root. Python 3.12, `.venv`.

**Train run 5.** `RunConfig` defaults in `src/run_final_training.py` *are* the run
5 spec; edit `main()` for a different output directory. It refuses to overwrite an
existing `final.pt`/`results.json`.

```powershell
.venv\Scripts\python src\run_final_training.py
```

~3h16m on GPU. Writes `checkpoints_run5/final.pt` and `results.json`.

**ID eval** (control step inside the OOD script, or standalone):

```powershell
.venv\Scripts\python src\run_ood_eval.py    # asserts ID == 836/1000, then runs OOD
```

~20s for the ID control, ~2min for the 3,000-graph OOD set on CPU.

**Diagnostics** (all CPU, all deterministic, all write to `results/`):

```powershell
.venv\Scripts\python diagnostics\m9b_eval.py                   # -> results\m9b_run5.txt          (~5 min)
.venv\Scripts\python diagnostics\diagnostic_length_vs_nodes.py # -> results\comb_and_size_run5.txt
.venv\Scripts\python diagnostics\diagnostic_ood.py             # -> results\ood_diagnostics_run5.txt
.venv\Scripts\python diagnostics\breakdown_run5.py             # -> results\breakdown_run5.txt
.venv\Scripts\python diagnostics\n50_extra_block.py            # -> results\n50_extra_block.txt
```

All five write their own output file (`m9b_eval.py` and `n50_extra_block.py` via
an in-script `Tee`/`emit`; the others print to stdout and were captured with
`tee`). No shell redirect is needed for the file to appear, which is why the
original M9b run command shows none.

`m9b_eval.py` was re-run end-to-end on 2026-09-22 with its output redirected to a
scratch path, and the result is **byte-for-byte identical** to the committed
`results/m9b_run5.txt` (all 203 lines, `diff` empty). Greedy decoding on CPU is
deterministic, so every figure quoted from that file is reproducible from the
checkpoint on disk.

`m9b_eval.py`, `diagnostic_length_vs_nodes.py` and `n50_extra_block.py` hard-code
`checkpoints_run5/final.pt` and their output filenames. `breakdown_run5.py` had
its `_SRC` and `CKPT_PATH` paths corrected on 2026-09-22 — it was written for the
repo root and broke when relocated into `diagnostics/`, failing with
`ModuleNotFoundError: No module named 'data'`. It now runs and its three sanity
counts (836 / 30 / 952) match `checkpoints_run5/results.json` exactly.

`n50_extra_block.py` re-measures the disputed N=50, E=60 configuration on a
disjoint seed range (30,000,000+) using the identical generator and evaluation
calls as `m9b_eval.py`, so the three N=50 blocks are directly comparable.

**Demo.** The Streamlit UI in `ui/app.py` talks to the FastAPI service in
`src/api.py` on port 8000:

```powershell
.venv\Scripts\python -m uvicorn src.api:app --port 8000
.venv\Scripts\python -m streamlit run ui\app.py
```

`requirements.txt` covers the training stack only. The demo additionally needs
`fastapi`, `uvicorn`, `streamlit`, `pyvis`, `plotly` and `pandas`, which are not
pinned there. `requirements.txt` and `run.log` are UTF-16LE-encoded (PowerShell
`Tee-Object` default); convert before piping into `pip install -r` on a POSIX
shell.

**Raw outputs.** `results/ood_run5.txt` (M9 OOD),
`results/ood_diagnostics_run5.txt` (decode-level diagnostics + N=18–30 sweep),
`results/length_vs_nodes_run5.txt` and `results/comb_and_size_run5.txt`
(length/size separation), `results/m9b_run5.txt` (fixed-length size sweep +
edge-count controls), `results/breakdown_run5.txt` (ID breakdown, failure
taxonomy, weight sensitivity under both definitions),
`results/n50_extra_block.txt` (third N=50 block, n=1000).

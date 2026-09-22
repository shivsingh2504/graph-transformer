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

*Added 2026-09-22, after this amendment was committed:* the hash above was
quoted without its recipe, so it was not reproducible from the repo. The recipe
now lives in `diagnostics/verify_data_hashes.py`, which recomputes both this
graph-level hash and the tokenisation hash of the same 200 graphs, prints
MATCH/MISMATCH for each, and exits non-zero on mismatch. Verified both ways:
exit 0 against the recorded values, exit 1 when the expected value is tampered
with. The hash itself is unchanged by this addition.

**Gate.** Amendment 1 waived the >=90% gate for run 5 only. Run 6 is subject to
the gate again: it passes if ID valid_and_optimal >= 90%.

**Authorized by:** project owner on 2026-09-22, who delegated the choice of
variable ("ur call") to the reviewer.

### Process exception 1: lock committed with the code it justifies, 2026-09-22
Amendment 2 and `src/run_training_run6.py` landed in the *same* commit
(`d154c74`), so the lock was written to justify code that already existed rather
than preceding it. The project rule is the reverse: the decision is locked, then
the code implements it.

A second, related deviation: the stored plan had run 6 as the `[SRC]`/`[DST]`
query-block move. Epoch count was substituted on new evidence (runs 4 vs 5 hold
the distribution and the 500k split fixed and differ only in epochs: 46.3% ->
83.6%) after the owner delegated the choice of variable. That substitution was
never ratified against the locked-decisions record before the commit went out.

Both were caught by review before any GPU time was spent, and the substance of
Amendment 2 was verified post-hoc: `RunConfig` at `d154c74` is field-for-field
identical to `fb37f42`, `tokenizer.py` / `model/` / `train/` / `run_experiment.py`
have zero diff lines, and the 30 -> 60 comes from a single
`dataclasses.replace(..., n_epochs=60)` in the new entry point, which asserts at
startup that `n_epochs` is the only differing field.

Recorded as an exception, not smoothed over. **Going forward the lock precedes
the commit, without exception.** The run 7 ruling below follows that order: it is
committed as a proposal, and no tracked code implementing it exists.

**Flagged by:** reviewer, 2026-09-22. Not treated as a blocker for run 6.

## Lock rulings

### PROPOSED - not ratified: run 7 encoder query-block position
Drafted 2026-09-22 while run 6 trains. **This is a proposal. It changes nothing
until ratified, and `src/data/tokenizer.py` is unmodified** - `encode_graph` still
emits the query block last. What does exist is validation code, tracked so every
figure below is reproducible from a clean checkout:

- `diagnostics/run7_layout_probe.py` - a standalone query-first encoder used only
  to execute pre-launch items 1 and 4. It is not wired into any pipeline, dataset
  or training path; it builds the new order from the tokenizer's own
  `_canonical_edges` / `_node_token_id` / `_weight_token_id` helpers so ordering is
  the only difference under test.
- `diagnostics/run7_layout_probe_negative.py` - deliberately broken variants that
  prove the probe's assertions can fail. Run this before trusting the probe.
- `diagnostics/check_run7_ruling_claims.py` - asserts all 71 numeric claims in this
  ruling against `results/m9b_run5.txt` and `results/ood_diagnostics_run5.txt`.
- `diagnostics/verify_data_hashes.py` - reproduces both reference hashes.

All four exit 0 from the repo root as of 2026-09-22. Recorded here so run 7 can
launch without another round-trip to spec it after the fact.

**The change.** `GraphTokenizer.encode_graph` (src/data/tokenizer.py:71) emits
the query block first:

```
[SRC] node_s [DST] node_t | e1u e1v e1w | e2u e2v e2w | ... | eEu eEv eEw
```

instead of the current layout, where the same four tokens trail the edge block
at absolute positions `3E` and `3E+2`. Edge content, canonical ordering
(`_canonical_edges`: `(min(u,v), max(u,v), w)`, sorted) and the decoder target
(`encode_path` = `[BOS] path [EOS]`) are unchanged. Total length stays `3E + 4`.

**Why - and how far the evidence actually goes.** Training covers N=5..20, and
under the locked rule `E = min(3N, N(N-1)/2)` that is exactly sixteen `(E, length)`
pairs, `length = 3E + 4`:

```
E    : 10  15  21  24  27  30  33  36  39  42  45  48  51  54  57  60
len  : 34  49  67  76  85  94 103 112 121 130 139 148 157 166 175 184
```

Everything the model ever saw lies on that line, and the longest input is **184
tokens**. `diagnostics/verify_data_hashes.py` measures min/max encoder length
34/184 over the first 200 real training graphs, so the bounds are observed, not
just derived from the formula.

What the evidence in results/m9b_run5.txt supports, and what it does not:

- **Supported, and it is the clean control.** The size sweep holds E=60 and
  length=184 fixed - both trained - and raises N from 25 to 50.
  `correct_endpoints` stays at 79.5-95.8% (V&O 26.2-63.0%). **Node count on its
  own does not cause the endpoint collapse.** At its own locked `(E=75, len=229)`,
  N=25 scores 0.5% V&O against 63.0% at length 184: a 126x gap at identical node
  count (results/ood_diagnostics_run5.txt:171).
- **Supported but confounded.** At N=20, every row inside the trained region
  (E=45..60, lengths 139..184) scores `correct_endpoints` 94.2-98.5%, and every
  row outside it (E=66/69/72, lengths 202/211/220) scores 3.0% / 0.2% / 0.2%.
  Because `length = 3E + 4`, E and length move together by construction, so this
  separates "inside the trained region" from "outside it" - it does **not**
  separate length from edge count. An earlier draft of this ruling claimed it
  isolated length. That was wrong and is corrected here.
- **Supported.** At the locked rule, N=21 is the first size whose pair
  `(E=63, len=193)` falls outside the trained region, and that is exactly where
  the cliff is: `correct_endpoints` 97.5% (N=20) -> 53.0% (N=21) -> 4.0% (N=22)
  (results/ood_diagnostics_run5.txt:166-168).

**Known open risk - the mechanism is correlational and incomplete.** Read this
before reading run 7's results against it.

Test 1 (results/m9b_run5.txt, N=7) breaks any simple version of the story. Its
four rows sit at lengths 40 / 43 / 58 / 61 and E = 12 / 13 / 18 / 19. **None of
those lengths is trained, and none of those E values is trained** - the only
trained pair at N=7 is `(21, 67)`. Yet `correct_endpoints` is 76.5% / 87.0% /
99.2% / **0.0%**. So "outside the trained region" does not predict collapse
either, and an earlier sentence in this ruling - "`correct_endpoints` is high at
every length <= 184 and collapses at every length > 184" - is **false as written**
and has been removed. Length <= 184 is at best necessary, not sufficient.

The 0.0% row (E=19, length 61) is not separated from its 99.2% neighbour (E=18,
length 58) by anything on record: length (61 vs 58, both untrained), marker
position (57/59 vs 54/56, both untrained - trained marker positions are
{30, 45, 63, 72, ..., 180}), density (0.905 vs 0.857), or mean Dijkstra hops
(1.46 vs 1.52). Divisibility of E by 3 looks tempting - E=18 is a multiple, E=19
is not, and the file itself labels them "Unseen Multiple" and "Contrast" - but the
other pair in the same test contradicts it: E=12 (multiple of 3) scores 76.5%
while E=13 (not a multiple) scores **87.0%**, the wrong way round. So that
reading cannot be adopted either. This row needs its own diagnostic before any
mechanism claim rests on it.

**Consequence for run 7.** The justification for the change is therefore *not*
"a length threshold explains the collapse". It is narrower and still real: the
size sweep shows node count is not the cause; the query block currently sits at
absolute positions `3E` and `3E+2`, which move with every change in E and N; and
pinning it at positions 0-3 removes that dependency at zero cost to sequence
length. If the true mechanism is something else entirely - edge-count
familiarity, density, or whatever separates E=18 from E=19 - run 7 will show it
as a criterion miss, which is why the success criterion below is fixed in advance
rather than interpreted afterwards.

**What this change is NOT expected to fix.** Moving the query block does not
shorten the sequence. At N=50 the edge block still occupies positions 4..453,
none of which were seen in training. And `edges_valid` decays with node count
even at fixed length 184 - 80.8% (N=25) -> 63.5% -> 51.5% -> 41.2% -> 34.0% ->
35.2% (N=50) - so picking valid edges out of a larger node set is a separate
failure that a position change cannot touch. The honest expectation is that run 7
repairs `correct_endpoints` at N > 20 and leaves OOD V&O well short of useful.

**Success criterion, fixed in advance.** Run 7 is a success on its own terms if,
at the locked edge rule for N=21..30, `correct_endpoints` rises from the run 5
baseline (53.0% / 4.0% / 1.5% / 0.0% / 0.5% / 0.0% / 0.5% / 0.0% for
N=21/22/23/24/25/26/28/30, results/ood_diagnostics_run5.txt:167-174) to >= 90% at
every one of those sizes, with ID valid_and_optimal not regressing more than 2
points from run 6. It is explicitly not a claim that OOD V&O becomes useful;
`edges_valid` is expected to remain the binding constraint there.

**Pipeline impact - searched, not assumed.** No other module encodes the layout:
- `collate_fn` (src/train/train.py:60) right-pads to `max_src` derived from the
  batch; it holds no fixed offsets.
- `_make_masks` (src/train/train.py:97) builds `src_mask` from pad ids and the
  causal mask from `tgt_len`; cross-attention mask is `(B, 1, 1, src_len)`,
  shape unchanged.
- `PositionalEncoding(max_len=5000)` (src/model/layers.py) already covers the
  454-token maximum at N=50.
- The only length arithmetic in the repo is `tok_len = 3 * num_edges + 4` in
  diagnostics/m9b_eval.py:67, diagnostics/diagnostic_length_vs_nodes.py:47 and
  diagnostics/n50_extra_block.py:60. The move preserves length, so these hold.
- Every `[0]`/`[-1]` endpoint check (src/eval/evaluate.py:153,
  src/diagnostic.py:106 and :176, diagnostics/diagnostic_ood.py:104 and :112)
  indexes the **decoded path**, not the encoder input.

**Existing tests that pin the old layout.** `test_src_dst_markers_at_end`
(tests/test_tokenizer.py:101) asserts `encoded[-4] == src_token_id` and
`encoded[-2] == dst_token_id`; it must be replaced by a front-of-sequence
equivalent, not deleted. `test_src_and_target_nodes_follow_markers` (:91) locates
the markers with `.index()` and survives unchanged, which makes it a useful
independent check that the markers still precede their node tokens.
`test_output_length_matches_formula` (:76) survives unchanged.

**Required before launch.** Items 1 and 4 were executed locally on 2026-09-22
against the `diagnostics/run7_layout_probe.py` prototype, before ratification, so
they are recorded as results rather than intentions. They must still be re-run
against the real `encode_graph` once it exists - a prototype passing is not the
implementation passing. Items 2, 3 and 5 remain to be done.

1. **DONE (prototype).** Permutation invariants over 1,850 real generated graphs
   in four splits - ID N=5-20 (1,000), OOD N=21-50 (600), fixed N=7 (200), fixed
   N=50 (50). For every graph: the first four ids are exactly
   `[src_token_id, node_s, dst_token_id, node_t]`; `new[4:] == old[:-4]`;
   `len(new) == len(old) == 3E + 4`; `Counter(new) == Counter(old)`; and
   `pad_token_id` never appears. Zero violations in all four splits; observed
   lengths 34-184 (ID), 193-454 (OOD), 67 (N=7), 454 (N=50). The old layout was
   confirmed to have the markers at the end, so the move is real and not a no-op.

   **Finding from the negative tests - do not weaken this item.** Four assertions
   sound redundant but are not. Deliberately broken variants were run through
   them:
   - dropping the `[DST]` marker: caught by all four (head, perm, length, Counter).
   - emitting `[DST] t [SRC] s` instead of `[SRC] s [DST] t`: caught by the head
     check **only**. `new[4:] == old[:-4]` compares the *edge* block, so any bug
     confined to the query block is invisible to it, and `Counter` is order-blind
     by construction. perm=0, len=0, counter=0, head=200.
   - swapping the query *nodes* while keeping the markers in place
     (`[SRC] target [DST] source`): again caught by the head check alone. This is
     the dangerous variant - it silently asks the model the wrong question while
     length, multiset and edge-block order all stay correct.

   So the explicit head assertion is load-bearing, not belt-and-braces. Item 1
   must keep all four, and the head assertion must compare against
   `graph.source` / `graph.target` directly rather than against the old sequence.

2. Labels must be untouched: Dijkstra `cost` and `path` for the probe split must
   be byte-identical to run 6's. Only the encoder input changes.

3. Recompute and record both reference hashes on the new format, using
   `diagnostics/verify_data_hashes.py`. The run 6 values become invalid for run 7
   and must not be reused:
   graphs `bf1adf1930c5a1d8d542b91319fca6b261c7109e8fb740fc55b73c1683fd9278`,
   tokenisation `0d7c7fdbfad17308703084ec400687f1e85426c5df07c3f547de9f1a256d0c88`.
   The graph hash is expected to be *unchanged* - run 7 does not touch generation
   or labelling - so a differing graph hash means the change leaked outside the
   tokenizer. The tokenisation hash is expected to change.

4. **DONE (prototype).** A 64-example batch with 16 distinct E values was pushed
   through the real `collate_fn` and `_make_masks` (src/train/train.py:60 and :97),
   for both the new and the old layout. Asserted: `src` shape `(64, 184)`;
   cross-attention mask shape `(64, 1, 1, 184)`; `src_mask == (src != pad)` exactly,
   so it flags pads and nothing else; per row, every real token unmasked, every
   pad masked, and the tail is all pad; for the new layout the first four columns
   are the query block in all 64 rows and stay unmasked even in the shortest row;
   and no real token equals `pad_id` (which is what makes the mask safe at all -
   `[PAD]`=0, node tokens start at 5). A left-padding collate was run as the
   negative test and was caught by the row, head and mask assertions.

   Note `create_padding_mask` (src/model/layers.py:11) returns `seq != pad`, i.e.
   True means *keep*. Anything asserting on mask polarity must get that the right
   way round.

5. Single variable. Run 7 changes only `encode_graph`, branches from the run 6
   commit, and carries its own amendment. `n_epochs` is set explicitly at that
   point from run 6's outcome rather than inherited by accident.

**Explicitly not decided here.** Whether run 7 also relaxes `E = 3N` so edge
count varies within a graph size. That is a second variable and needs its own
ruling. Test 1 is the reason to consider it: at fixed N=7, varying E over
12/13/18/19 moves `correct_endpoints` across 76.5% / 87.0% / 99.2% / 0.0%, so the
model is clearly sensitive to edge count in a way the locked rule never exposed it
to. But the sensitivity is not monotone and not explained (see the open risk
above), so it cannot ride along with the position change on the strength of a
pattern nobody understands yet.

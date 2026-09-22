# Bugfix Requirements: Query Block Fixed Positions

## Introduction

The graph-transformer shortest-path model suffers from severe out-of-distribution (OOD) accuracy collapse and in-distribution (ID) performance plateauing below the 90% valid-and-optimal gate. The root cause is that the encoder's positional encoding of source/destination query markers depends on the number of edges (E), causing the model to encounter token positions during evaluation that were never seen during training. This document specifies the requirements to fix this by moving the query block to fixed token positions (0-3), independent of the edge block size.

## Bug Analysis

### Current Behavior (Defect)

**Current Layout (Buggy):**
```
[edge block (3E tokens)] ... [SRC src DST tgt (4 tokens)]
Marker positions: 3E, 3E+2 (position-dependent on E)
```

1.1 WHEN input length exceeds training maximum (184 tokens) due to E > 61 THEN marker tokens appear at positions never seen in training, causing severe OOD accuracy collapse

1.2 WHEN out-of-distribution graphs have N=25-50 (E=75-150 tokens) with locked E=3N THEN model achieves 0.0% endpoint accuracy despite sufficient total sequence length

1.3 WHEN input sequences have identical length but different edge counts (e.g., both 184 tokens with E=45-60 vs E=61-63) THEN endpoint accuracy shows discontinuous degradation (65-72% vs 0-4.5%) unrelated to sequence length alone

1.4 WHEN in-distribution evaluation occurs THEN model plateaus at 83.6%-86.1% valid-and-optimal accuracy, failing to reach the 90% acceptable gate

### Expected Behavior (Correct)

**Proposed Layout (Fixed):**
```
[SRC src DST tgt (4 tokens)] [edge block (3E tokens)]
Marker positions: 0, 2 (always fixed, independent of E)
```

2.1 WHEN query block is moved to fixed positions 0-3 THEN marker tokens appear at identical positions across all training examples, enabling the encoder to learn consistent positional representations

2.2 WHEN out-of-distribution graphs have N=25-50 (E=75-150) THEN encoder can extrapolate positional patterns beyond training length, preventing position-induced representation collapse

2.3 WHEN endpoint accuracy is measured on OOD graphs with N=21-30 THEN the system SHALL achieve >= 90% endpoint accuracy (primary success metric)

2.4 WHEN query block is fixed at positions 0-3 THEN marker token representations become stable and generalizable across sequence lengths

### Unchanged Behavior (Regression Prevention)

3.1 WHEN in-distribution evaluation occurs on run 7 with the new query block layout THEN the system SHALL CONTINUE TO maintain valid-and-optimal accuracy within -2 percentage points of run 6's 86.1% (i.e., >= 84.1%)

3.2 WHEN valid-structure metrics are evaluated on out-of-distribution graphs THEN the system SHALL CONTINUE TO maintain edges_valid >= 80% (no regression in edge validity beyond current OOD decay pattern)

3.3 WHEN in-distribution graphs are encoded THEN the system SHALL CONTINUE TO produce correct graph structure tokens and edge adjacencies

3.4 WHEN tokenizer processes different graph sizes (N=5-20 in training) THEN the system SHALL CONTINUE TO produce consistent token sequences for the fixed query block regardless of N

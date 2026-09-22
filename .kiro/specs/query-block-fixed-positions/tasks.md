# Implementation Plan: Query Block Fixed Positions

## Workflow: Correctness & Backward Compatibility Verification BEFORE Training

This plan prioritizes **implementation correctness** and **backward compatibility validation** before committing 6.5 hours to training. The workflow verifies the fix is safe and effective with high confidence.

---

## Phase 1: Code Changes (Tokenizer + Tests) — PROCEED NOW

- [ ] 1.1 Implement query block reordering in encode_graph
  - **File**: `src/data/tokenizer.py`, method `encode_graph(self, graph: Graph) -> List[int]`
  - **Current Code** (lines 59-67):
    ```python
    ids : List[int] = []
    for (u, v, w) in self._canonical_edges(graph):
      ids.append(self._node_token_id(u))
      ids.append(self._node_token_id(v))
      ids.append(self._weight_token_id(w))
    
    ids.append(self.src_token_id)
    ids.append(self._node_token_id(graph.source))
    ids.append(self.dst_token_id)
    ids.append(self._node_token_id(graph.target))
    return ids
    ```
  - **Changes**:
    1. Move query block assembly to BEGINNING of method (before edge loop)
    2. Implementation pattern:
       ```python
       ids : List[int] = []
       # Query block first (positions 0-3)
       ids.append(self.src_token_id)
       ids.append(self._node_token_id(graph.source))
       ids.append(self.dst_token_id)
       ids.append(self._node_token_id(graph.target))
       
       # Edge block follows (positions 4 onward)
       for (u, v, w) in self._canonical_edges(graph):
           ids.append(self._node_token_id(u))
           ids.append(self._node_token_id(v))
           ids.append(self._weight_token_id(w))
       
       return ids
       ```
    3. Do NOT modify: `_canonical_edges()`, `_node_token_id()`, `_weight_token_id()`, `encode_path()`, `decode_path()`, `_build_vocab()`
  - _Bug_Condition: Current query block at [-(4), -(3), -(2), -(1)]_
  - _Expected_Behavior: Fixed query block at [0, 1, 2, 3]_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 1.2 Update test assertions for new query block position
  - **File**: `tests/test_tokenizer.py`, class `TestEncodeGraph`
  - **Test 1 - Modify `test_src_dst_markers_at_end`** (lines 101-103):
    - OLD:
      ```python
      assert encoded[-4] == tokenizer.src_token_id
      assert encoded[-2] == tokenizer.dst_token_id
      ```
    - NEW:
      ```python
      assert encoded[0] == tokenizer.src_token_id
      assert encoded[2] == tokenizer.dst_token_id
      ```
    - Rename test to `test_src_dst_markers_at_front`
  - **Test 2 - Modify `test_src_and_target_nodes_follow_markers`** (lines 91-98):
    - OLD:
      ```python
      src_pos = encoded.index(tokenizer.src_token_id)
      dst_pos = encoded.index(tokenizer.dst_token_id)
      assert encoded[src_pos + 1] == tokenizer.token_to_id[f"node_{simple_graph.source}"]
      assert encoded[dst_pos + 1] == tokenizer.token_to_id[f"node_{simple_graph.target}"]
      ```
    - NEW (query block at fixed [0, 1, 2, 3]):
      ```python
      assert encoded[0] == tokenizer.src_token_id
      assert encoded[1] == tokenizer.token_to_id[f"node_{simple_graph.source}"]
      assert encoded[2] == tokenizer.dst_token_id
      assert encoded[3] == tokenizer.token_to_id[f"node_{simple_graph.target}"]
      ```
  - _Expected_Behavior: Query block now at [0, 1, 2, 3]_
  - _Requirements: 2.1, 2.3_

- [ ] 1.3 Add new test case for query block at front
  - **File**: `tests/test_tokenizer.py`, add to class `TestEncodeGraph`
  - **New Test**: `test_query_block_at_front`
    ```python
    def test_query_block_at_front(self, tokenizer: GraphTokenizer, simple_graph: Graph) -> None:
        """Query block (SRC, src_node, DST, dst_node) is at positions 0-3."""
        encoded = tokenizer.encode_graph(simple_graph)
        # Positions 0-3: query block
        assert encoded[0] == tokenizer.src_token_id
        assert encoded[1] == tokenizer._node_token_id(simple_graph.source)
        assert encoded[2] == tokenizer.dst_token_id
        assert encoded[3] == tokenizer._node_token_id(simple_graph.target)
        
        # Edge block follows
        assert len(encoded) == 3 * len(simple_graph.edges) + 4
    ```
  - _Expected_Behavior: Query block at [0, 1, 2, 3]_
  - _Requirements: 2.1, 2.3_

---

## Phase 2: Correctness Verification — PROCEED NOW

- [ ] 2.1 Run all tokenizer unit tests
  - **Command**: `pytest tests/test_tokenizer.py -v --tb=short`
  - **Expected**: All tests pass including:
    - ✓ `test_output_length_matches_formula` (unchanged, 3E+4 still valid)
    - ✓ `test_src_dst_markers_at_front` (modified, now checks positions [0, 2])
    - ✓ `test_src_and_target_nodes_follow_markers` (modified, now checks [0, 1, 2, 3])
    - ✓ `test_query_block_at_front` (new test)
    - ✓ All other tests (unchanged)
  - **Success**: All unit tests pass on fixed code
  - _Requirements: 2.1, 2.3, 3.1, 3.2_

- [ ] 2.2 Verify permutation property (markers, edge block unchanged)
  - **Test Goal**: Confirm query block moved, edge block unchanged
  - **Test Implementation**:
    ```python
    def test_permutation_property():
        """Verify markers are at front, edge block follows unchanged."""
        tokenizer = GraphTokenizer()
        graphs = [
            Graph(num_nodes=5, edges=[(0,1,1), (1,2,2), (2,3,1)], source=0, target=3, seed=42),
            Graph(num_nodes=10, edges=[(i, i+1, 1) for i in range(9)], source=0, target=9, seed=99)
        ]
        
        for graph in graphs:
            encoded = tokenizer.encode_graph(graph)
            
            # Verify query block at [0, 1, 2, 3]
            assert encoded[0] == tokenizer.src_token_id, f"Query block not at front"
            assert encoded[2] == tokenizer.dst_token_id, f"Query block not at front"
            
            # Verify edge block starts at position 4
            assert encoded[4] in [tokenizer.token_to_id[f"node_{i}"] for i in range(50)], \
                f"Edge block doesn't start at position 4"
            
            # Verify total length unchanged
            assert len(encoded) == 3 * len(graph.edges) + 4, \
                f"Length formula violated: {len(encoded)} != {3 * len(graph.edges) + 4}"
    ```
  - **Run**: Execute test
  - **Expected**: Test passes, confirms markers at 0/2, edge block unchanged
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 2.3 Multiset preservation check (via Counter)
  - **Test Goal**: Verify all tokens present before and after, no additions/removals
  - **Test Implementation**:
    ```python
    from collections import Counter
    
    def test_multiset_preserved():
        """Verify token multiset is identical (permutation proof)."""
        tokenizer_fixed = GraphTokenizer()
        
        graphs = [
            Graph(num_nodes=5, edges=[(0,1,1), (1,2,2), (2,3,1)], source=0, target=3, seed=42),
        ]
        
        for graph in graphs:
            encoded = tokenizer_fixed.encode_graph(graph)
            
            # Manually reconstruct old layout (edges + query block at end)
            old_layout = []
            for (u, v, w) in tokenizer_fixed._canonical_edges(graph):
                old_layout.append(tokenizer_fixed._node_token_id(u))
                old_layout.append(tokenizer_fixed._node_token_id(v))
                old_layout.append(tokenizer_fixed._weight_token_id(w))
            old_layout.append(tokenizer_fixed.src_token_id)
            old_layout.append(tokenizer_fixed._node_token_id(graph.source))
            old_layout.append(tokenizer_fixed.dst_token_id)
            old_layout.append(tokenizer_fixed._node_token_id(graph.target))
            
            # Verify multiset equality
            assert Counter(encoded) == Counter(old_layout), \
                f"Multiset mismatch: {Counter(encoded)} vs {Counter(old_layout)}"
    ```
  - **Run**: Execute test
  - **Expected**: Counter equal, confirms permutation (no token additions/removals)
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 2.4 Length invariant verification (formula 3E+4)
  - **Test Goal**: Verify length stays 3E+4 for all graph sizes
  - **Test Implementation**:
    ```python
    def test_length_invariant():
        """Verify length formula 3E + 4 holds for all graphs."""
        tokenizer = GraphTokenizer()
        
        for num_edges in [1, 5, 10, 20, 50]:
            edges = [(i, i+1, (i % 9) + 1) for i in range(num_edges)]
            graph = Graph(num_nodes=num_edges+1, edges=edges, source=0, target=num_edges, seed=42)
            
            encoded = tokenizer.encode_graph(graph)
            expected_length = 3 * num_edges + 4
            
            assert len(encoded) == expected_length, \
                f"E={num_edges}: len={len(encoded)}, expected={expected_length}"
    ```
  - **Run**: Execute test
  - **Expected**: Length equals 3E+4 for all E values
  - _Requirements: 2.1, 3.1, 3.2_

---

## Phase 3: Backward Compatibility Checks — PROCEED NOW (Critical)

- [ ] 3.1 Padding and right-alignment verification
  - **Test Goal**: Confirm collate_fn pads correctly with new layout
  - **Test Implementation**: Create batch with varying lengths, verify padding applied to right only
  - **Expected**: Padding tokens only at end of sequence, no interference with query block
  - _Requirements: 3.1, 3.2_

- [ ] 3.2 Attention mask verification (covers only padding)
  - **Test Goal**: Verify masks generated for new layout cover only padding
  - **Expected**: Masks correctly identify padding, query block [0-3] not masked
  - _Requirements: 3.1, 3.2_

- [ ] 3.3 Collate_fn shape verification
  - **Test Goal**: Verify batch collation produces expected tensor shapes
  - **Expected**: Batch shape correct, query block at same positions across all rows
  - _Requirements: 3.1, 3.2_

- [ ] 3.4 Downstream code check (no position assumptions)
  - **Analysis**:
    1. Search for `encoded[-4]`, `encoded[-3]`, `encoded[-2]`, `encoded[-1]` in codebase
    2. Search for `.index(tokenizer.src_token_id)` in production code (not tests)
    3. Verify cross-attention logic doesn't assume query block position
    4. Verify decoder inference doesn't depend on marker positions
  - **Expected**: No hard-coded position assumptions in production code
  - _Requirements: 3.1, 3.2_

- [ ] 3.5 Test against existing cached tokenized sequences
  - **Analysis**:
    1. Check if `datasets/` contains cached .pt or .pkl files
    2. Verify run 7 will regenerate sequences (don't load stale cache)
    3. Confirm no code loads old cached sequences
  - **Expected**: Caches regenerated; no stale sequence issues
  - _Requirements: 3.1, 3.2_

---

## Phase 4: Checkpoint Infrastructure Validation — PROCEED NOW

- [ ] 4.1 Verify training script checkpoint save/resume
  - **File**: `src/run_training_run6.py`
  - **Verification**: Review checkpoint save/resume logic, verify unchanged with new layout
  - **Expected**: Checkpoint save/resume works unchanged
  - _Requirements: 3.1_

- [ ] 4.2 Verify checkpoint path resolution
  - **File**: `src/run_training_run6.py`
  - **Verification**: Check checkpoint directory refs, verify path resolution works
  - **Expected**: Paths resolve correctly
  - _Requirements: 3.1_

- [ ] 4.3 Verify epoch/loss tracking unchanged
  - **File**: `src/run_training_run6.py`, `src/train/`
  - **Verification**: Review training loop metrics, verify unaffected by tokenizer
  - **Expected**: Metrics work unchanged
  - _Requirements: 3.1_

- [ ] 4.4 Verify reproducibility (seed → identical data → identical hashes)
  - **Test Goal**: Confirm reproducibility with new layout
  - **Expected**: Same seed produces identical sequences, hashes match
  - _Requirements: 3.1_

---

## Phase 5: Decision Gate — AFTER PHASE 4 COMPLETES

- [ ] 5.1 Gate Assessment
  - **Success Criteria (ALL must pass)**:
    - ✅ Phase 1: Code changes implemented correctly
    - ✅ Phase 2: All unit tests pass (correctness verified)
    - ✅ Phase 3: Backward compatibility checks pass (no regressions)
    - ✅ Phase 4: Checkpoint infrastructure works (reproducibility verified)
  
  - **Decision Logic**:
    - If ALL pass → **Proceed to Phase 6 (Optional Diagnostic)**
    - If ANY fails → **Stop, diagnose, fix, re-validate**
  
  - **After Gate**: High confidence fix is correct, ready to train

---

## Phase 6: Optional - Quick Diagnostic Before Full Training — CONDITIONAL

- [ ] 6.1 Optional: Run 10-minute layout diagnostic (if additional confidence needed)
  - **Goal**: Quick sanity check before committing 6.5 hours
  - **Test**: Generate 100 random graphs across E values, verify layout properties hold
  - **Expected**: Diagnostic passes, layout properties confirmed
  - **Decision**: 
    - If passes → **Proceed to Phase 7 (Training)**
    - If fails → **Investigate, fix, retest**
  - _Requirements: 2.1, 2.2, 2.3_

---

## Phase 7: Training Run 7 — AFTER GATE PASSES

- [ ] 7.1 Create training script for run 7
  - **File**: Create `src/run_training_run7.py`
  - **Changes**:
    1. Copy `src/run_training_run6.py`
    2. Change checkpoint dir: `checkpoints_run6/` → `checkpoints_run7/`
    3. Change run ID: `run_6` → `run_7`
    4. Inherit all other settings
  - **Expected**: Script ready to execute

- [ ] 7.2 Execute training run 7
  - **Command**: `python src/run_training_run7.py`
  - **Expected Duration**: ~6.5 hours on GPU
  - **Expected Outputs**:
    - `checkpoints_run7/latest.pt`
    - `checkpoints_run7/final.pt`
    - `checkpoints_run7/results.json`
  - **Success**: Training completes, checkpoints saved

---

## Phase 8: Evaluation — AFTER RUN 7 COMPLETES

- [ ] 8.1 Run ID evaluation control
  - **Goal**: Confirm -2pt max regression vs run 6's 86.1%
  - **Expected**: >= 84.1% valid-and-optimal accuracy
  - **Success**: ID gate maintained

- [ ] 8.2 Run OOD evaluation
  - **Goal**: Measure OOD improvement with new layout
  - **Expected**: >= 90% endpoint accuracy (primary success metric)
  - **Success**: OOD gate achieved

- [ ] 8.3 Breakdown analysis by size
  - **Goal**: Understand improvement distribution
  - **Expected**: Improvement across all OOD sizes, no unexpected regressions
  - **Success**: Results validated

---

## Success Criteria Before Training

✅ All tokenizer unit tests pass (existing + new)  
✅ Query block permutation verified (markers at 0/2, edge block unchanged)  
✅ Multiset and length invariants confirmed  
✅ Padding/masking work correctly with new layout  
✅ Collate_fn produces correct batch shapes  
✅ No downstream code breaks  
✅ Checkpoint save/resume works unchanged  
✅ Reproducibility verified (seed → identical outputs)  
✅ Decision gate passed  
✅ **Ready to train with high confidence**


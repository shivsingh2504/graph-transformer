# Query-Block-Fixed-Positions Bugfix Design

## Overview

This bugfix addresses a layout inefficiency in the graph tokenizer's sequence encoding. Currently, the source and target markers (`[SRC]`, `[DST]`) and their corresponding node IDs are appended to the end of the canonical edge block, requiring the decoder to skip past all E edges (3E tokens) before accessing the query information needed to guide path decoding.

By moving the query block (source and target nodes) to the beginning of the sequence, the decoder gains immediate access to the starting and ending constraints, improving architectural alignment without changing the total sequence length (3E + 4 tokens remain constant). This is a pure permutation: the token count and vocab are unchanged; only the order is different.

The fix maintains full backward compatibility at the pipeline level—collate_fn, mask generation, and decoder inference work unchanged with any layout—so only the tokenizer's encode_graph method needs modification, plus two test assertions that pin absolute positions.

## Glossary

- **Bug_Condition (C)**: The condition where the tokenizer's encode_graph method places the query block (source and target node IDs) at the end of the sequence instead of at the beginning
- **Property (P)**: The desired behavior where the query block is positioned at the front of the sequence (positions 0–3) for immediate decoder access
- **Preservation**: Existing functionality of non-graph-encoding pipeline stages (collate_fn, mask generation, decoder inference) that must remain unchanged
- **encode_graph**: The method in `src/data/tokenizer.py` that converts a Graph into a list of token IDs with the edge block and query block
- **Canonical edges**: The sorted, deduplicated edge list from a graph, used to ensure deterministic ordering regardless of input order
- **Query block**: The four-token sequence `[SRC] node_source [DST] node_target` representing the source and target nodes as fixed query constraints
- **Edge block**: The sequence of 3E tokens representing the E canonical edges as (u, v, weight) triples
- **Sequence layout**: The ordering of query block and edge block; current layout places edges first, new layout places query block first
- **Total sequence length**: 3E + 4, fixed by design, where E is the number of edges

## Bug Details

### Bug Condition

The bug manifests when the `encode_graph` method constructs a sequence for a graph with E edges. Instead of placing the query block (source and target nodes with their markers) at the beginning where the decoder would access them first, the method appends the query block to the end of the canonical edge block, forcing the decoder to skip 3E edge tokens before reaching the source and target constraints.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type Graph with E edges
  OUTPUT: boolean
  
  RETURN TRUE if encode_graph places [SRC, src_node, DST, dst_node]
         at positions [-(4), -(3), -(2), -(1)] 
         instead of at positions [0, 1, 2, 3]
END FUNCTION
```

In the current implementation:
- `encode_graph` builds the sequence as: `[edge_0_u, edge_0_v, edge_0_w, ..., edge_E-1_u, edge_E-1_v, edge_E-1_w, SRC, src_node, DST, dst_node]`
- This places the query block at indices [-(4), -(3), -(2), -(1)] (i.e., the last 4 positions)
- The decoder must process all edges before accessing source/target information

### Examples

**Example 1: Small graph (E=1)**
- Graph: 1 edge (0→1, weight=3), source=0, target=1
- Current output: `[node_0, node_1, weight_3, SRC, node_0, DST, node_1]` (length 7)
  - Positions 0–2: edge block
  - Positions 3–6: query block
  - Query block is at the end
- Expected output: `[SRC, node_0, DST, node_1, node_0, node_1, weight_3]` (length 7)
  - Positions 0–3: query block
  - Positions 4–6: edge block
  - Query block is at the front
- Formula holds: 3×1 + 4 = 7 ✓

**Example 2: Larger graph (E=4)**
- Graph: 4 canonical edges, source=0, target=3
- Current output: 16 tokens total (3×4 + 4)
  - First 12 tokens: edge block
  - Last 4 tokens: `[SRC, node_0, DST, node_3]`
- Expected output: 16 tokens total (3×4 + 4)
  - First 4 tokens: `[SRC, node_0, DST, node_3]`
  - Next 12 tokens: edge block
- Same length, different order; pure permutation

**Example 3: Out-of-distribution graph (E=50, N=50)**
- Graph: 50 canonical edges, source=5, target=47
- Current: 154 tokens (3×50 + 4), query block at positions [150, 151, 152, 153]
- Fixed: 154 tokens, query block at positions [0, 1, 2, 3]
- This ensures the decoder always finds source/target info at predictable positions, regardless of graph size

**Edge case: Query node IDs match edge node IDs**
- Even when source/target appear in the edge block, the query block must still be moved to the front
- Example: source=0, target=1, and edges include (0, 1)
- The tokens appear in both places (edge block AND query block), but their semantic roles are different
- Moving preserves both occurrences without loss of information

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
1. The total sequence length remains 3E + 4 for any graph with E edges
2. The complete token multiset is preserved (no tokens added or removed, only reordered)
3. Both source and target nodes appear exactly where the decoder expects them
4. All edge tokens (u, v, w) appear in the same canonical order
5. The special tokens [SRC] and [DST] remain in the sequence at fixed known positions
6. No padding tokens are introduced by the tokenizer
7. Token ID generation (vocab mapping) remains identical
8. encode_path() behavior is completely unchanged
9. decode_path() behavior is completely unchanged
10. Graph validation and edge canonicalization logic remain unchanged

**Scope:**
All code that does NOT directly call encode_graph should be unaffected:
- Mask generation (_make_masks) works with any layout because it only checks for pad tokens
- Collate_fn pads sequences and stacks them but doesn't care about token order
- Decoder inference uses generated token IDs and vocabulary indices, not absolute positions in the encoder output
- Cross-attention uses the full encoder sequence but doesn't depend on query block position
- Evaluation metrics (decode_path, path cost calculation) work on decoder output, not encoder layout

**Testing Approach:**
Property-based testing confirms that for all graphs where the bug condition does NOT hold (i.e., after the fix), the multiset of tokens equals the original and total length is preserved.

## Hypothesized Root Cause

The bug is not a defect in the traditional sense (crash, wrong output for non-buggy inputs). Rather, it's a design suboptimality: the tokenizer was originally implemented by appending markers and nodes to the end of the edge block, a natural sequential construction. However, this layout is architecturally misaligned with the decoder's inference flow:

1. **Original Construction Pattern**: Developers wrote a loop to add edges, then sequentially appended the source/target markers—a straightforward "build all content, add metadata at the end" approach
2. **Decoder Consumption Mismatch**: The decoder processes the encoder output left-to-right, but the query constraints (source/target) are at the tail, requiring the model to learn to look past all edges to find them
3. **Cross-Attention Inefficiency**: When the decoder attends to the encoder, forcing it to skip 3E tokens before reaching the query block may reduce attention efficiency and increase training complexity
4. **Layout Optimization Opportunity**: By placing the query block first (a pure permutation), we align the layout with the natural decoder inference pattern without any cost in sequence length

**Why This Is Not a Bug in the Traditional Sense:**
- The current implementation is functionally correct (no crashes, wrong outputs, or data corruption)
- The fix is optional from a correctness standpoint, but strategic from a performance/architecture standpoint
- Run 7 trains from scratch with the new layout, allowing the model to learn the improved layout without the need to adapt from old positional patterns

## Correctness Properties

Property 1: Query Block Position - Reordered Query Markers

_For any_ graph input where the bug condition holds (current layout has query block at the end), the fixed encode_graph function SHALL place the source and target markers and nodes at positions 0–3 of the output sequence: `[SRC, src_node, DST, dst_node, ...]`.

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation - Sequence Multiset and Length Unchanged

_For any_ graph input, the fixed encode_graph function SHALL produce output with:
- Identical total length (3E + 4)
- Identical multiset of tokens (same tokens in same frequencies, only order changes)
- All edge tokens in canonical order
- No introduction of pad tokens or vocab collisions

This ensures backward compatibility with pipeline stages (collate_fn, masks, decoder) that do not depend on absolute token positions.

**Validates: Requirements 3.1, 3.2, 3.3**

## Fix Implementation

### Changes Required

**File**: `src/data/tokenizer.py`

**Method**: `encode_graph(self, graph: Graph) -> List[int]`

**Specific Changes**:

1. **Reorder Construction**: Move query block assembly to the beginning of the method
   - Instead of appending SRC/dst markers to the end, place them first
   - The method will now construct: `[SRC, src_node, DST, dst_node]` before the edge block

2. **Edge Block Assembly**: Keep edge block logic unchanged, but now it follows the query block
   - Loop through canonical edges as before: `(u, v, w) → node_u, node_v, weight_w`
   - Append to the query block instead of to an empty list

3. **Implementation Pattern**:
   ```
   ids = []
   # Query block: place first
   ids.append(self.src_token_id)
   ids.append(self._node_token_id(graph.source))
   ids.append(self.dst_token_id)
   ids.append(self._node_token_id(graph.target))
   
   # Edge block: follow the query block
   for (u, v, w) in self._canonical_edges(graph):
       ids.append(self._node_token_id(u))
       ids.append(self._node_token_id(v))
       ids.append(self._weight_token_id(w))
   
   return ids
   ```

4. **No Changes to Helper Methods**: `_canonical_edges()`, `_node_token_id()`, `_weight_token_id()` remain untouched

5. **No Changes to Other Methods**: `encode_path()`, `decode_path()`, `_build_vocab()` all remain untouched

### Why This Is Safe

- **Pure Permutation**: Only the order changes; no new tokens, no removed tokens, no vocab changes
- **No Cross-Method Dependencies**: encode_path and decode_path work on the output, not on internal logic
- **Batch Processing Tolerant**: Collate_fn and mask generation only need pad masking and shape consistency, not positional knowledge
- **One-Line-at-a-Time Construction**: The change is purely in the order of list appends; no algorithmic change
- **Determinism Preserved**: Same input graph produces same tokens in same relative order, just repositioned

## Testing Strategy

### Validation Approach

The testing strategy uses the bug condition methodology:
1. **Exploratory Bug Condition Checking**: Confirm the bug condition holds on unfixed code (query block at end)
2. **Fix Checking**: Verify the fixed code places query block at the front for all test graphs
3. **Preservation Checking**: Verify sequence multiset and length remain unchanged, and pipeline stages work correctly

### Exploratory Bug Condition Checking

**Goal**: Surface examples of the current buggy layout BEFORE implementing the fix. Confirm the bug condition via the existing test suite and diagnostic tools.

**Test Plan**: Run the existing `run7_layout_probe.py` diagnostic WITHOUT modifying the tokenizer to confirm:
- Current layout has query block at positions [-(4), -(3), -(2), -(1)]
- New layout in the diagnostic function places query block at positions [0, 1, 2, 3]
- The two are permutations of each other

**Test Cases**:
1. **Layout Observation Test**: Inspect encode_graph output to confirm query block is at the end
2. **Permutation Verification Test**: Run layout_probe on unfixed tokenizer to show the diagnostic's reordered version differs from current tokenizer output
3. **Mixed-E Batch Test**: Generate a batch with varying E values and confirm all rows have query block at different positions in current layout

**Expected Counterexamples**:
- Graphs with E=1 (short sequence): query block at indices [3, 4, 5, 6]
- Graphs with E=20 (long sequence): query block at indices [56, 57, 58, 59]
- The diagnostic's reordered version has query block at indices [0, 1, 2, 3] for all graphs

### Fix Checking

**Goal**: Verify that after the fix, all graphs have their query block at the front.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := encode_graph_fixed(input)
  ASSERT result[0] == SRC
  ASSERT result[1] == node_source
  ASSERT result[2] == DST
  ASSERT result[3] == node_target
END FOR
```

**Unit Tests to Add**:
1. **Query Block at Front Test**: Assert query block occupies positions 0–3 after fix
2. **Edge Block Follows Test**: Assert edge block (canonical order) occupies positions 4 onward
3. **Small Graph Test**: E=1, verify fixed layout
4. **Large Graph Test**: E=50, verify fixed layout
5. **Boundary Test**: E=49 (max typical), verify fixed layout

### Preservation Checking

**Goal**: Verify that the multiset and length of tokens remain identical, and pipeline stages work unchanged.

**Pseudocode:**
```
FOR ALL input DO
  old := encode_graph_original(input)
  new := encode_graph_fixed(input)
  ASSERT len(new) == len(old)
  ASSERT Counter(new) == Counter(old)
  ASSERT set(new) == set(old)
END FOR
```

**Test Plan**: Use property-based testing to generate random graphs and verify:
- For any valid graph, multiset is preserved
- For any valid graph, length formula (3E + 4) holds
- Batch processing (collate_fn + masks) produces valid tensors
- Decoder can infer paths using either layout (via integration test)

**Test Cases**:
1. **Multiset Preservation Test**: Compare Counter(old) == Counter(new) for random graphs
2. **Length Preservation Test**: Verify len(new) == 3*E + 4 for any E
3. **Batch Masking Test**: Process fixed-layout batch through collate_fn and _make_masks, confirm masks cover only pads
4. **Collate Query Block Test**: Verify first four columns of collated batch contain query block for all rows in new layout
5. **End-to-End Batch Test**: Run new layout through collate_fn, mask generation, and cross-attention calculation; verify no shape mismatches or out-of-bounds errors

### Unit Tests

These tests modify existing assertions in `tests/test_tokenizer.py`:

- `test_output_length_matches_formula`: Unchanged, formula 3E+4 still holds
- `test_src_dst_markers_present_in_output`: Unchanged, markers still present
- `test_src_and_target_nodes_follow_markers`: MODIFIED to check positions 1 and 3 instead of src_pos+1, dst_pos+1
- `test_src_dst_markers_at_end`: MODIFIED assertion to check positions [0, 1, 2, 3] instead of [-(4), -(3), -(2), -(1)]
- `test_query_block_at_front`: NEW test asserting query block at front after fix

### Property-Based Tests

Use Hypothesis or similar PBT framework:

- **Property**: For any graph with E edges, `Counter(encode_graph(g)) == Counter(original_layout(g))`
  - Generates random node counts, edge counts, and source/target pairs
  - Verifies multiset invariant across hundreds of random graphs
  
- **Property**: For any graph, `len(encode_graph(g)) == 3 * len(g.edges) + 4`
  - Confirms length formula holds for all E in [1, 50]
  
- **Property**: For any graph, first four tokens are `[SRC, src_node, DST, dst_node]`
  - Verifies query block position for all graphs
  - Catches off-by-one errors in position calculation

### Integration Tests

- **Batch Processing Test**: Create a mixed-E batch with fixed layout, process through collate_fn, confirm shape and masking work correctly
- **Decoder Inference Test**: Run a small model through inference using the new layout, verify path decoding produces valid results
- **Full Training Loop (Run 7)**: Train from scratch using the new layout for a small number of steps; confirm no crashes, losses decrease, and final model can infer paths

## Summary

This bugfix implements a pure permutation of the tokenizer's encode_graph output, moving the query block (source and target nodes) from the end to the front of the sequence. The change:

- Improves architectural alignment (decoder sees query constraints first)
- Maintains total sequence length (3E + 4, unchanged)
- Preserves all tokens (multiset identical)
- Works unchanged with all pipeline stages (collate_fn, masks, decoder)
- Requires only tokenizer modification plus two test assertion updates
- Is validated by existing layout_probe diagnostic and new property-based tests

Run 7 trains from scratch with the new layout, avoiding the need to adapt pretrained positional embeddings.

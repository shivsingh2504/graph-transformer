"""
Phase 3: Backward Compatibility Tests
Verify that the query block reordering doesn't break downstream code
"""
import pytest
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.graph_generator import Graph, generate_random_connected_graph
from data.tokenizer import GraphTokenizer


@pytest.fixture
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


class TestPaddingAndAlignment:
    """Phase 3.1: Padding and right-alignment verification"""

    def test_padding_applied_only_to_end(self, tokenizer: GraphTokenizer) -> None:
        """Verify padding tokens only appear at the end of sequence."""
        # Create a simple graph
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        encoded = tokenizer.encode_graph(graph)
        
        # Simulate padding to length 30
        max_length = 30
        padded = encoded + [tokenizer.pad_token_id] * (max_length - len(encoded))
        
        # Verify padding only at the end
        num_encoded = len(encoded)
        for i in range(num_encoded):
            assert padded[i] != tokenizer.pad_token_id, \
                f"Pad token found at position {i}, should only be at end"
        
        for i in range(num_encoded, max_length):
            assert padded[i] == tokenizer.pad_token_id, \
                f"Non-pad token at position {i} (should be padding)"

    def test_query_block_not_padded(self, tokenizer: GraphTokenizer) -> None:
        """Verify query block (positions 0-3) is never padded."""
        for seed in range(5):
            graph = generate_random_connected_graph(
                num_nodes=10,
                seed=seed,
                edge_density=0.4,
                min_weight=1,
                max_weight=10,
            )
            encoded = tokenizer.encode_graph(graph)
            
            # Verify query block positions 0-3 are not pad tokens
            for pos in range(4):
                assert encoded[pos] != tokenizer.pad_token_id, \
                    f"Seed {seed}: Query block position {pos} contains pad token"


class TestMaskGeneration:
    """Phase 3.2: Attention mask verification (covers only padding)"""

    def test_mask_covers_only_padding(self, tokenizer: GraphTokenizer) -> None:
        """Verify that masks should only mask padding tokens, not query block."""
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        encoded = tokenizer.encode_graph(graph)
        
        # Simulate padded sequence
        max_length = 30
        padded = encoded + [tokenizer.pad_token_id] * (max_length - len(encoded))
        
        # Create mask: 1 for valid tokens, 0 for padding
        mask = [1 if token != tokenizer.pad_token_id else 0 for token in padded]
        
        # Verify query block (positions 0-3) is not masked
        for pos in range(4):
            assert mask[pos] == 1, f"Query block position {pos} is masked (should be unmasked)"
        
        # Verify only padding is masked
        num_encoded = len(encoded)
        for i in range(num_encoded):
            assert mask[i] == 1, f"Valid token at position {i} is masked"
        
        for i in range(num_encoded, max_length):
            assert mask[i] == 0, f"Padding at position {i} is not masked"

    def test_query_block_attention_accessible(self, tokenizer: GraphTokenizer) -> None:
        """Verify query block is always accessible for attention."""
        graphs = [
            Graph(
                num_nodes=5,
                edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
                source=0,
                target=3,
                seed=42,
            ),
            generate_random_connected_graph(
                num_nodes=15,
                seed=99,
                edge_density=0.3,
                min_weight=1,
                max_weight=10,
            ),
        ]
        
        for graph in graphs:
            encoded = tokenizer.encode_graph(graph)
            padded = encoded + [tokenizer.pad_token_id] * (50 - len(encoded))
            
            # Mask for any valid token (including query block)
            attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in padded]
            
            # Query block positions should have attention_mask == 1
            for pos in range(4):
                assert attention_mask[pos] == 1, \
                    f"Query block position {pos} not accessible to attention"


class TestBatchingAndShapes:
    """Phase 3.3: Collate_fn shape verification"""

    def test_batch_shape_consistency(self, tokenizer: GraphTokenizer) -> None:
        """Verify batch stacking produces correct shapes."""
        # Create a batch of graphs with varying edge counts
        graphs = [
            generate_random_connected_graph(
                num_nodes=n,
                seed=s,
                edge_density=0.3,
                min_weight=1,
                max_weight=10,
            )
            for s, n in enumerate([5, 8, 10, 12], start=42)
        ]
        
        # Encode all graphs
        encoded_list = [tokenizer.encode_graph(g) for g in graphs]
        
        # Pad to max length in batch
        max_len = max(len(seq) for seq in encoded_list)
        padded_list = [
            seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            for seq in encoded_list
        ]
        
        # Convert to tensor (simulating batch collation)
        batch_tensor = torch.tensor(padded_list, dtype=torch.long)
        
        # Verify shape
        assert batch_tensor.shape[0] == len(graphs), \
            f"Batch size mismatch: {batch_tensor.shape[0]} vs {len(graphs)}"
        assert batch_tensor.shape[1] == max_len, \
            f"Sequence length mismatch: {batch_tensor.shape[1]} vs {max_len}"
        
        # Verify query block is at consistent positions across batch
        for batch_idx in range(len(graphs)):
            assert batch_tensor[batch_idx, 0] == tokenizer.src_token_id, \
                f"Batch {batch_idx}: SRC token not at position 0"
            assert batch_tensor[batch_idx, 2] == tokenizer.dst_token_id, \
                f"Batch {batch_idx}: DST token not at position 2"

    def test_query_block_alignment_in_batch(self, tokenizer: GraphTokenizer) -> None:
        """Verify query block is at same positions in all batch examples."""
        graphs = [
            generate_random_connected_graph(
                num_nodes=10,
                seed=s,
                edge_density=0.4,
                min_weight=1,
                max_weight=10,
            )
            for s in range(10)
        ]
        
        encoded_list = [tokenizer.encode_graph(g) for g in graphs]
        max_len = max(len(seq) for seq in encoded_list)
        padded_list = [
            seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            for seq in encoded_list
        ]
        
        batch_tensor = torch.tensor(padded_list, dtype=torch.long)
        
        # All batch rows should have query block at positions [0, 1, 2, 3]
        for batch_idx in range(len(graphs)):
            assert batch_tensor[batch_idx, 0].item() == tokenizer.src_token_id
            assert batch_tensor[batch_idx, 2].item() == tokenizer.dst_token_id
            # Position 1 and 3 should be node tokens (not pad, not special markers)
            node_token_ids = {tokenizer.token_to_id[f"node_{n}"] for n in range(50)}
            assert batch_tensor[batch_idx, 1].item() in node_token_ids
            assert batch_tensor[batch_idx, 3].item() in node_token_ids


class TestDownstreamCodeAssumptions:
    """Phase 3.4: Downstream code check for position assumptions"""

    def test_no_hard_coded_position_assumptions(self, tokenizer: GraphTokenizer) -> None:
        """Verify encoded sequences work regardless of using position indices directly."""
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        encoded = tokenizer.encode_graph(graph)
        
        # The sequence should work with any code that:
        # 1. Only cares about padding (not absolute positions)
        # 2. Uses .index() to find markers instead of hard-coded indices
        # 3. Passes full sequences to attention mechanisms
        
        # Verify markers can be found (not assuming fixed position)
        src_idx = encoded.index(tokenizer.src_token_id)
        dst_idx = encoded.index(tokenizer.dst_token_id)
        
        assert src_idx == 0, "SRC marker not at position 0 (may break assumptions)"
        assert dst_idx == 2, "DST marker not at position 2 (may break assumptions)"
        
        # Verify source/target nodes follow their markers
        assert encoded[src_idx + 1] == tokenizer._node_token_id(graph.source)
        assert encoded[dst_idx + 1] == tokenizer._node_token_id(graph.target)

    def test_full_sequence_immutability(self, tokenizer: GraphTokenizer) -> None:
        """Verify the complete token sequence is passed through unchanged."""
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        encoded = tokenizer.encode_graph(graph)
        
        # All tokens in the sequence should be valid
        for token_id in encoded:
            assert 0 <= token_id < tokenizer.vocab_size, \
                f"Invalid token ID {token_id} (vocab size: {tokenizer.vocab_size})"
            
            # Should be in token_to_id mapping
            token_name = tokenizer.id_to_token.get(token_id)
            assert token_name is not None, f"Token ID {token_id} not in reverse mapping"


class TestCachedSequences:
    """Phase 3.5: Test against cached sequences (reproducibility)"""

    def test_deterministic_encoding(self, tokenizer: GraphTokenizer) -> None:
        """Verify same graph always produces identical encoding."""
        graph = Graph(
            num_nodes=5,
            edges=[(0, 1, 1), (1, 2, 2), (2, 3, 1)],
            source=0,
            target=3,
            seed=42,
        )
        
        # Encode multiple times
        encodings = [tokenizer.encode_graph(graph) for _ in range(10)]
        
        # All should be identical
        first = encodings[0]
        for i, encoding in enumerate(encodings[1:], 1):
            assert encoding == first, \
                f"Encoding {i} differs from first encoding"

    def test_seed_determines_sequence(self, tokenizer: GraphTokenizer) -> None:
        """Verify same seed produces reproducible sequences."""
        graphs_same_seed = [
            generate_random_connected_graph(
                num_nodes=10,
                seed=42,
                edge_density=0.4,
                min_weight=1,
                max_weight=10,
            )
            for _ in range(3)
        ]
        
        encodings = [tokenizer.encode_graph(g) for g in graphs_same_seed]
        
        # All should be identical (same seed = same graph)
        first = encodings[0]
        for i, encoding in enumerate(encodings[1:], 1):
            assert encoding == first, \
                f"Seed 42, encoding {i} differs from first"

    def test_different_seeds_produce_different_sequences(
        self, tokenizer: GraphTokenizer
    ) -> None:
        """Verify different seeds produce different sequences."""
        graphs = [
            generate_random_connected_graph(
                num_nodes=10,
                seed=s,
                edge_density=0.4,
                min_weight=1,
                max_weight=10,
            )
            for s in range(10)
        ]
        
        encodings = [tokenizer.encode_graph(g) for g in graphs]
        
        # Count unique encodings (should be close to len(graphs))
        unique_encodings = len(set(tuple(enc) for enc in encodings))
        
        # Most should be unique (not all - small chance of collision)
        assert unique_encodings >= 8, \
            f"Expected mostly unique encodings, got {unique_encodings}/10"

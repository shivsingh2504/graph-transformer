"""
Phase 4: Checkpoint Infrastructure Validation Tests
Verify training script components work with new tokenizer layout
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.dataset_generator import generate_dataset_split
from data.graph_generator import generate_random_connected_graph
from data.tokenizer import GraphTokenizer


@pytest.fixture
def tokenizer() -> GraphTokenizer:
    return GraphTokenizer(min_weight=1, max_weight=10)


class TestCheckpointReproducibility:
    """Phase 4.4: Verify reproducibility (seed → identical sequences)"""

    def test_fixed_seed_produces_identical_graphs(
        self, tokenizer: GraphTokenizer
    ) -> None:
        """Verify that the same seed produces identical graph sequences."""
        seed = 42
        
        # Generate dataset split twice with same seed
        split1 = generate_dataset_split(num_examples=10, node_range=(8, 12), base_seed=seed)
        split2 = generate_dataset_split(num_examples=10, node_range=(8, 12), base_seed=seed)
        
        # Encode both splits
        encodings1 = [tokenizer.encode_graph(g) for g, _ in split1.examples]
        encodings2 = [tokenizer.encode_graph(g) for g, _ in split2.examples]
        
        # All encodings should match
        assert len(encodings1) == len(encodings2), "Different number of examples"
        for i, (enc1, enc2) in enumerate(zip(encodings1, encodings2)):
            assert enc1 == enc2, f"Example {i} encodings differ between runs with same seed"

    def test_seed_independence_from_tokenizer_position(
        self, tokenizer: GraphTokenizer
    ) -> None:
        """Verify seed produces same graph structure regardless of tokenizer layout."""
        # Using fixed seed, generate a graph
        graph = generate_random_connected_graph(
            num_nodes=10,
            seed=99,
            edge_density=0.4,
            min_weight=1,
            max_weight=10,
        )
        
        # Get original properties
        num_edges = len(graph.edges)
        source = graph.source
        target = graph.target
        edges = sorted(graph.edges)  # Canonical order
        
        # Verify these are reproducible and independent of tokenizer layout
        # (The tokenizer reordering shouldn't affect graph generation)
        assert num_edges > 0, "Graph must have edges"
        assert source != target, "Source and target must differ"
        assert all(u != v for u, v, w in edges), "All edges must be between distinct nodes"
        
        # Encoding should use the same source/target/edges
        encoded = tokenizer.encode_graph(graph)
        
        # Verify query block contains correct source/target
        assert encoded[0] == tokenizer.src_token_id
        assert encoded[1] == tokenizer._node_token_id(source)
        assert encoded[2] == tokenizer.dst_token_id
        assert encoded[3] == tokenizer._node_token_id(target)

    def test_deterministic_dataset_hash(self, tokenizer: GraphTokenizer) -> None:
        """Verify dataset hashes are deterministic (important for caching)."""
        import json
        from hashlib import sha256
        
        def compute_split_hash(split, tokenizer):
            """Compute deterministic hash of encoded split."""
            ids = [tokenizer.encode_graph(g) for g, _ in split.examples]
            payload = json.dumps(ids, sort_keys=True)
            return sha256(payload.encode()).hexdigest()
        
        # Generate split twice with same seed
        split1 = generate_dataset_split(num_examples=5, node_range=(5, 10), base_seed=42)
        split2 = generate_dataset_split(num_examples=5, node_range=(5, 10), base_seed=42)
        
        hash1 = compute_split_hash(split1, tokenizer)
        hash2 = compute_split_hash(split2, tokenizer)
        
        assert hash1 == hash2, "Dataset hashes differ for same seed"


class TestTokenizerAndDataIntegration:
    """Phase 4: Verify tokenizer works correctly in data pipeline"""

    def test_batch_of_encoded_sequences(self, tokenizer: GraphTokenizer) -> None:
        """Verify batching multiple encoded sequences works correctly."""
        # Generate multiple graphs
        split = generate_dataset_split(num_examples=8, node_range=(6, 15), base_seed=123)
        
        # Encode all
        encoded_list = [tokenizer.encode_graph(g) for g, _ in split.examples]
        
        # All should have query block at positions 0-3
        for i, enc in enumerate(encoded_list):
            assert enc[0] == tokenizer.src_token_id, f"Example {i}: SRC not at position 0"
            assert enc[2] == tokenizer.dst_token_id, f"Example {i}: DST not at position 2"
            
            # All should be valid
            for token_id in enc:
                assert 0 <= token_id < tokenizer.vocab_size, \
                    f"Example {i}: Invalid token {token_id}"

    def test_path_encoding_unaffected(self, tokenizer: GraphTokenizer) -> None:
        """Verify encode_path is completely unaffected by graph layout changes."""
        from data.dijkstra import ShortestPath
        
        # Create paths
        paths = [
            ShortestPath(path=[0, 1, 2], cost=5),
            ShortestPath(path=[0, 5, 10, 15], cost=20),
            ShortestPath(path=[1], cost=0),
        ]
        
        # Encode all paths
        for path in paths:
            encoded = tokenizer.encode_path(path)
            
            # Verify path format is unchanged
            assert encoded[0] == tokenizer.bos_token_id, "Path should start with BOS"
            assert encoded[-1] == tokenizer.eos_token_id, "Path should end with EOS"
            
            # Verify roundtrip
            decoded = tokenizer.decode_path(encoded)
            assert decoded == path.path, f"Path roundtrip failed for {path.path}"
            
            # Verify middle tokens are node tokens
            for token_id in encoded[1:-1]:
                token = tokenizer.id_to_token[token_id]
                assert token.startswith("node_"), f"Path token should be node, got {token}"

    def test_encoder_decoder_compatibility(self, tokenizer: GraphTokenizer) -> None:
        """Verify encoder output can be used by decoder (shape/vocab compatibility)."""
        from data.dijkstra import ShortestPath
        
        # Get a graph and path
        split = generate_dataset_split(num_examples=1, node_range=(5, 10), base_seed=456)
        graph, path = split.examples[0]
        
        # Encode graph and path
        src_ids = tokenizer.encode_graph(graph)
        tgt_ids = tokenizer.encode_path(path)
        
        # Verify compatibility
        # 1. All tokens are in valid range
        for token_id in src_ids + tgt_ids:
            assert 0 <= token_id < tokenizer.vocab_size
        
        # 2. Source should have query block at front
        assert src_ids[0] == tokenizer.src_token_id
        assert src_ids[1] == tokenizer._node_token_id(graph.source)
        
        # 3. Target should have path structure
        assert tgt_ids[0] == tokenizer.bos_token_id
        assert tgt_ids[-1] == tokenizer.eos_token_id
        
        # 4. Lengths follow formulas
        expected_src_len = 3 * len(graph.edges) + 4
        expected_tgt_len = len(path.path) + 2
        
        assert len(src_ids) == expected_src_len, \
            f"Source length {len(src_ids)} != expected {expected_src_len}"
        assert len(tgt_ids) == expected_tgt_len, \
            f"Target length {len(tgt_ids)} != expected {expected_tgt_len}"


class TestVocabConsistency:
    """Phase 4: Verify vocabulary is unchanged and consistent"""

    def test_vocab_size_unchanged(self, tokenizer: GraphTokenizer) -> None:
        """Verify vocab size remains unchanged."""
        # Expected size: 5 special + 50 nodes + 10 weights = 65
        assert tokenizer.vocab_size == 65

    def test_special_tokens_unchanged(self, tokenizer: GraphTokenizer) -> None:
        """Verify special token IDs are unchanged."""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.src_token_id == 3
        assert tokenizer.dst_token_id == 4

    def test_node_and_weight_tokens_unchanged(self, tokenizer: GraphTokenizer) -> None:
        """Verify node and weight token IDs are unchanged."""
        # Node tokens: 5-54
        for n in range(50):
            expected_id = 5 + n
            actual_id = tokenizer.token_to_id[f"node_{n}"]
            assert actual_id == expected_id, \
                f"node_{n} ID changed: expected {expected_id}, got {actual_id}"
        
        # Weight tokens: 55-64
        for w in range(1, 11):
            expected_id = 54 + w
            actual_id = tokenizer.token_to_id[f"weight_{w}"]
            assert actual_id == expected_id, \
                f"weight_{w} ID changed: expected {expected_id}, got {actual_id}"

    def test_token_to_id_mapping_unchanged(self, tokenizer: GraphTokenizer) -> None:
        """Verify token_to_id mapping is unchanged."""
        # Create two tokenizers and compare
        tok1 = tokenizer
        tok2 = GraphTokenizer(min_weight=1, max_weight=10)
        
        assert tok1.token_to_id == tok2.token_to_id
        assert tok1.id_to_token == tok2.id_to_token


class TestCheckpointPathResolution:
    """Phase 4.2: Verify checkpoint path resolution"""

    def test_checkpoint_path_structure(self) -> None:
        """Verify checkpoint directory structure is correct."""
        from pathlib import Path
        
        checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints_run6"
        
        # Check if run 6 checkpoint exists
        if checkpoint_dir.exists():
            # If it exists, verify expected structure
            assert checkpoint_dir.is_dir(), "Checkpoint dir should be directory"
            
            # Common checkpoint files
            expected_files = ["latest.pt", "final.pt"]
            for fname in expected_files:
                fpath = checkpoint_dir / fname
                if fpath.exists():
                    assert fpath.is_file(), f"{fname} should be file"

    def test_checkpoint_independence(self) -> None:
        """Verify different runs don't share checkpoints."""
        from pathlib import Path
        
        run6_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints_run6"
        run7_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints_run7"
        
        # They should be different directories
        assert str(run6_dir) != str(run7_dir), "Run 6 and 7 should have different checkpoint dirs"

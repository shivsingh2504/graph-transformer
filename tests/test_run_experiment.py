from __future__ import annotations
 
import os
import sys
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import torch
import torch.nn as nn
import pytest
 
from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from model.model import Transformer
from train.train import _make_masks, _train_epoch, make_dataloader
from run_experiment import run_experiment
 
DEVICE = torch.device("cpu")
 
_TINY_TRAIN_KWARGS = dict(
    batch_size=4,
    lr=1e-3,
    warmup_steps=2,
    n_layers=1,
    d_model=32,
    n_heads=4,
    d_ff=64,
    dropout=0.0,
    device=DEVICE,
)
 
def _tiny_model(vocab_size: int) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    ).to(DEVICE)
 
 
def _global_grad_norm(model: Transformer) -> float:
    return torch.sqrt(
        sum(
            p.grad.detach().norm() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
    ).item()
 

class TestSeedValidation:
    def test_raises_when_seeds_equal(self) -> None:
        with pytest.raises(ValueError, match="train_seed and eval_seed must differ"):
            run_experiment(
                num_train_examples=4,
                num_eval_examples=4,
                n_epochs=1,
                train_seed=42,
                eval_seed=42,
                **_TINY_TRAIN_KWARGS,
            )
 
    def test_raises_when_n_epochs_in_kwargs(self) -> None:
        with pytest.raises(ValueError, match="'n_epochs' must be passed"):
            run_experiment(
                num_train_examples=4,
                num_eval_examples=4,
                n_epochs=1,
                train_seed=0,
                eval_seed=1,
                n_epochs=2,
                **{**_TINY_TRAIN_KWARGS, "n_epochs": 2},
            )
 
    def test_does_not_raise_when_seeds_differ(self) -> None:
        result = run_experiment(
            num_train_examples=4,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=0,
            eval_seed=1,
            **_TINY_TRAIN_KWARGS,
        )
        assert result is not None

class TestReturnedDictStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return run_experiment(
            num_train_examples=8,
            num_eval_examples=4,
            n_epochs=2,
            train_seed=10,
            eval_seed=20,
            **_TINY_TRAIN_KWARGS,
        )
 
    def test_top_level_keys_present(self, result) -> None:
        assert set(result.keys()) == {"train_loss", "val_loss", "eval_summary", "gate_passed"}
 
    def test_train_loss_is_list_of_floats(self, result) -> None:
        assert isinstance(result["train_loss"], list)
        assert all(isinstance(v, float) for v in result["train_loss"])
 
    def test_val_loss_is_list_of_floats(self, result) -> None:
        assert isinstance(result["val_loss"], list)
        assert all(isinstance(v, float) for v in result["val_loss"])
 
    def test_train_loss_length_matches_n_epochs(self, result) -> None:
        assert len(result["train_loss"]) == 2
 
    def test_val_loss_length_matches_n_epochs(self, result) -> None:
        assert len(result["val_loss"]) == 2
 
    def test_eval_summary_is_dict_of_floats(self, result) -> None:
        s = result["eval_summary"]
        assert isinstance(s, dict)
        assert all(isinstance(v, float) for v in s.values())
 
    def test_eval_summary_contains_valid_and_optimal_fraction(self, result) -> None:
        assert "valid_and_optimal_fraction" in result["eval_summary"]
 
    def test_gate_passed_is_bool(self, result) -> None:
        assert isinstance(result["gate_passed"], bool)

class TestTinyEndToEnd:
    def test_completes_without_error_and_fraction_in_unit_interval(self) -> None:
        result = run_experiment(
            num_train_examples=8,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=100,
            eval_seed=200,
            **_TINY_TRAIN_KWARGS,
        )
        vof = result["eval_summary"]["valid_and_optimal_fraction"]
        assert 0.0 <= vof <= 1.0, f"valid_and_optimal_fraction out of [0,1]: {vof}"
 
    def test_gate_passed_consistent_with_fraction(self) -> None:
        result = run_experiment(
            num_train_examples=8,
            num_eval_examples=4,
            n_epochs=1,
            train_seed=101,
            eval_seed=202,
            **_TINY_TRAIN_KWARGS,
        )
        vof = result["eval_summary"]["valid_and_optimal_fraction"]
        assert result["gate_passed"] == (vof >= 0.90)
 
    def test_losses_are_finite(self) -> None:
        result = run_experiment(
            num_train_examples=8,
            num_eval_examples=4,
            n_epochs=2,
            train_seed=102,
            eval_seed=203,
            **_TINY_TRAIN_KWARGS,
        )
        for loss_val in result["train_loss"] + result["val_loss"]:
            assert torch.isfinite(torch.tensor(loss_val)), (
                f"Non-finite loss encountered: {loss_val}"
            )
            
class TestSplitSeedIsolation:
    def test_first_graphs_differ_when_seeds_differ(self) -> None:
        train_split = generate_dataset_split(
            num_examples=4,
            node_range=(5, 20),
            base_seed=0,
        )
        eval_split = generate_dataset_split(
            num_examples=4,
            node_range=(5, 20),
            base_seed=1,
        )
        train_graph, _ = train_split.examples[0]
        eval_graph, _ = eval_split.examples[0]
 
        graphs_identical = (
            train_graph.num_nodes == eval_graph.num_nodes
            and sorted(train_graph.edges) == sorted(eval_graph.edges)
        )
        assert not graphs_identical, (
            "First graph in train split and eval split are identical even though "
            "different base_seeds were used — seed isolation is broken."
        )
 
    def test_same_seed_produces_same_first_graph(self) -> None:
        split_a = generate_dataset_split(
            num_examples=2, node_range=(5, 20), base_seed=7
        )
        split_b = generate_dataset_split(
            num_examples=2, node_range=(5, 20), base_seed=7
        )
        graph_a, _ = split_a.examples[0]
        graph_b, _ = split_b.examples[0]
        assert graph_a.num_nodes == graph_b.num_nodes
        assert sorted(graph_a.edges) == sorted(graph_b.edges)
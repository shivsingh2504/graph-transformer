from __future__ import annotations
 
from typing import Any, Dict
 
from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import EvalResult, evaluate_split
from train.train import train_model
 
_NODE_RANGE = (5, 20)
_SUCCESS_GATE: float = 0.90

def run_experiment(
    *,
    num_train_examples: int,
    num_eval_examples: int,
    n_epochs: int,
    train_seed: int,
    eval_seed: int,
    **train_model_kwargs: Any,
) -> Dict[str, Any]:
    if train_seed == eval_seed:
        raise ValueError(
            f"train_seed and eval_seed must differ to prevent split overlap, "
            f"but both are {train_seed}."
        )
 
    if "n_epochs" in train_model_kwargs:
        raise ValueError(
            "'n_epochs' must be passed as the dedicated 'n_epochs' argument "
            "to run_experiment(), not inside **train_model_kwargs."
        )
 
    train_split = generate_dataset_split(
        num_examples=num_train_examples,
        node_range=_NODE_RANGE,
        base_seed=train_seed,
    )
 
    eval_split = generate_dataset_split(
        num_examples=num_eval_examples,
        node_range=_NODE_RANGE,
        base_seed=eval_seed,
    )
 
    tokenizer = GraphTokenizer()
 
    model, history = train_model(
        train_split,
        eval_split,
        tokenizer,
        n_epochs=n_epochs,
        **train_model_kwargs,
    )
 
    eval_result: EvalResult = evaluate_split(model, eval_split, tokenizer)
 
    vof: float = eval_result.valid_and_optimal_fraction
 
    return {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "eval_summary": eval_result.summary(),
        "gate_passed": vof >= _SUCCESS_GATE,
    }
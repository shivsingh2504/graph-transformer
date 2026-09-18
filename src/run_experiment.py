from __future__ import annotations
 
from typing import Any, Dict
 
from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from eval.evaluate import EvalResult, evaluate_split
from train.train import train_model
 
_NODE_RANGE = (5, 20)
_SUCCESS_GATE: float = 0.90
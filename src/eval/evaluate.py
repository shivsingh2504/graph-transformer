from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
 
import torch
 
from data.dataset_generator import DatasetSplit
from data.dijkstra import ShortestPath
from data.graph_generator import Graph
from data.tokenizer import GraphTokenizer
from model.model import Transformer
 
_MAX_DECODE_LEN: int = 60

@dataclass(frozen=True)
class ExampleResult:
  decoded_nodes : List[int]
  edges_valid : bool
  valid_path : bool
  correct_endpoints:bool
  optimal_cost:bool
  valid_and_optimal : bool
  decoded_cost: Optional[int]
  true_cost : int
  
@dataclass
class EvalResult:
  per_example : List[ExampleResult] = field(default_factory=list)
  
  @property
  def num_examples(self)->int:
    return len(self.per_example)  
  
  @property
  def num_valid_and_optimal(self)->int:
    return sum(1 for r in self.per_example if r.valid_and_optimal)
  
  @property
  def valid_and_optimal_fraction(self)->float:
    if self.num_examples == 0:
      return 0.0
    return self.num_valid_and_optimal / self.num_examples
  
  @property
  def edges_valid_fraction(self)->float:
    if self.num_examples == 0:
      return 0.0
    return sum(1 for r in self.per_example if r.edges_valid) / self.num_examples
  
  @property
  def valid_path_fraction(self)->float:
    if self.num_examples == 0:
      return 0.0
    return sum(1 for r in self.per_example if r.edges_valid) / self.num_examples
  @property
  def correct_endpoints_fraction(self)->float:
    if self.num_examples == 0:
      return 0.0
    return sum(1 for r in self.per_example if r.edges_valid) / self.num_examples
  
  @property
  def optimal_cost_fraction(self)->float:
    if self.num_examples == 0:
      return 0.0
    return sum(1 for r in self.per_example if r.edges_valid) / self.num_examples
  
  def summary(self) -> Dict[str, float]:
    return {
      "valid_and_optimal_fraction": self.valid_and_optimal_fraction,
      "edges_valid_fraction": self.edges_valid_fraction,
      "valid_path_fraction": self.valid_path_fraction,
      "correct_endpoints_fraction": self.correct_endpoints_fraction,
      "optimal_cost_fraction": self.optimal_cost_fraction,
      "num_examples": float(self.num_examples),
      "num_valid_and_optimal": float(self.num_valid_and_optimal),
    }
    

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
  

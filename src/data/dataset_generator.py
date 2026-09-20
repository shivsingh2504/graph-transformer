from __future__ import annotations
import random
from data.graph_generator import Graph, generate_random_connected_graph
from data.dijkstra import ShortestPath, run_dijkstra
from typing import List,Tuple,Optional
from dataclasses import dataclass,field

_SEED_MULTIPLIER:int = 1_000_000
Example = Tuple[Graph,ShortestPath]

@dataclass
class DatasetSplit:
  examples : List[Example] = field(default_factory=list)
  num_examples : int = 0
  node_range: Tuple[int,int] = (2,2)
  base_seed : int = 0
  
  def __post_init__(self)->None:
    if self.num_examples != len(self.examples):
      raise ValueError(
                f"DatasetSplit.num_examples ({self.num_examples}) does not "
                f"match len(examples) ({len(self.examples)})."
      )

def generate_dataset_split(
  num_examples : int,
  node_range : Tuple[int,int],
  base_seed : int,
  *,
  num_edges : Optional[int] = None,
  edge_density : Optional[float] = None,
  min_weight :int =1,
  max_weight:int=10
)->DatasetSplit:
  if num_examples < 0:
    raise ValueError(f"num_examples must be >= 0, got {num_examples}.")
  if num_examples > _SEED_MULTIPLIER:
    raise ValueError(
            f"num_examples ({num_examples}) exceeds the seed scheme's "
            f"collision-free bound of {_SEED_MULTIPLIER}."
    )
  if base_seed < 0 or base_seed >= _SEED_MULTIPLIER:
    raise ValueError(
            f"base_seed must be in [0, {_SEED_MULTIPLIER}), got {base_seed}."
    )
  min_nodes , max_nodes = node_range
  if min_nodes < 2:
    raise ValueError(f"node_range[0] (min_nodes) must be >= 2, got {min_nodes}.")
  if max_nodes < min_nodes:
    raise ValueError(
            f"node_range[1] (max_nodes={max_nodes}) must be >= "
            f"node_range[0] (min_nodes={min_nodes})."
    )
  if num_edges is not None:
    max_edges : int = min_nodes*(min_nodes-1)//2
    if num_edges > max_edges:
      raise ValueError(
                f"num_edges={num_edges} is unsafe for node_range={node_range}: "
                f"min_nodes={min_nodes} allows at most {max_edges} edges. "
                f"Reduce num_edges to <= {max_edges}, raise min_nodes, "
                f"or use edge_density instead (it scales per graph automatically)."
      )
  examples : List[Example] = []
  for i in range(num_examples):
    per_ex_seed : int = base_seed * _SEED_MULTIPLIER + i
    node_rng = random.Random(per_ex_seed)
    num_nodes : int  = node_rng.randint(min_nodes,max_nodes)
    # Calculate dynamic edge count per Run 3 lock
    dynamic_edges = min(3 * num_nodes, num_nodes * (num_nodes - 1) // 2)
    
    graph : Graph = generate_random_connected_graph(
      num_edges=dynamic_edges,
      seed = per_ex_seed,
      num_nodes=num_nodes,
      min_weight=min_weight,
      max_weight=max_weight,
    )
    shortest_path : ShortestPath = run_dijkstra(graph)
    examples.append((graph, shortest_path))
  return DatasetSplit(
    examples=examples,
    num_examples=len(examples),
    node_range=node_range,
    base_seed=base_seed
  )



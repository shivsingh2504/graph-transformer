from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List,Tuple,Dict,Set,Optional

# Node ids are relabelled into the tokenizer's node vocabulary, so this caps N.
MAX_NODES: int = 50


def locked_edge_count(num_nodes: int) -> int:
    """Edge count mandated by the Run 3 lock: E = min(3N, N(N-1)/2)."""
    return min(3 * num_nodes, num_nodes * (num_nodes - 1) // 2)


@dataclass
class Graph:
  num_nodes : int
  edges : List[Tuple[int,int,int]]
  source : int
  target : int
  seed : int
  node_ids : List[int] = None
  
  def __post_init__(self):
      if self.node_ids is None:
          self.node_ids = list(range(self.num_nodes))
          
  def adjacency(self) -> Dict[int,List[Tuple[int,int]]]:
    adj : Dict[int,List[Tuple[int,int]]] = {i:[] for i in self.node_ids}
    for u,v,w in self.edges:
      adj[u].append((v,w))
      adj[v].append((u,w))
    return adj
  
  def to_dict(self)->dict:
    return{
      "num_nodes":self.num_nodes,
      "edges":[[u,v,w]for u,v,w in self.edges],
      "source":self.source,
      "target":self.target,
      "seed":self.seed,
      "node_ids": self.node_ids
    }

def _build_spanning_tree(node_order : List[int],rng:random.Random,min_weight:int,max_weight:int)-> Tuple[Set[Tuple[int, int]], List[Tuple[int, int, int]]]:
  edge_set: Set[Tuple[int,int]] = set()
  edges : List[Tuple[int,int,int]] = []
  connected: List[int] = [node_order[0]]
  for i in range(1,len(node_order)):
    new_node = node_order[i]
    existing = rng.choice(connected)
    u,v = min(new_node,existing),max(new_node,existing)
    w = rng.randint(min_weight,max_weight)
    edge_set.add((u,v))
    edges.append((u,v,w))
    connected.append(new_node)
    
  return edge_set,edges

def _add_extra_edges(num_nodes:int,edge_set:Set[Tuple[int, int]],edges: List[Tuple[int, int, int]],target_count: int,rng: random.Random,min_weight: int,max_weight: int)->None:
  candidates = [ (u,v) for u in range(num_nodes) for v in range(u+1,num_nodes) if (u,v) not in edge_set]
  rng.shuffle(candidates)
  for u,v in candidates:
    if len(edges) >= target_count:
      break
    w = rng.randint(min_weight,max_weight)
    edge_set.add((u,v))
    edges.append((u,v,w))

def _resolve_endpoints(
    num_nodes: int,
    source: Optional[int],
    target: Optional[int],
    rng: random.Random,
) -> Tuple[int, int]:
    if source is None and target is None:
        source, target = rng.sample(range(num_nodes), 2)
    elif source is None:
        candidates = [n for n in range(num_nodes) if n != target]
        source = rng.choice(candidates)
    elif target is None:
        candidates = [n for n in range(num_nodes) if n != source]
        target = rng.choice(candidates)

    return source, target

def _resolve_endpoints(
    num_nodes: int,
    source: Optional[int],
    target: Optional[int],
    rng: random.Random,
) -> Tuple[int, int]:
    if source is None and target is None:
        source, target = rng.sample(range(num_nodes), 2)
    elif source is None:
        candidates = [n for n in range(num_nodes) if n != target]
        source = rng.choice(candidates)
    elif target is None:
        candidates = [n for n in range(num_nodes) if n != source]
        target = rng.choice(candidates)

    return source, target


def generate_random_connected_graph(
    num_nodes: int,
    seed: int,
    num_edges: Optional[int] = None,
    edge_density: Optional[float] = None,
    min_weight: int = 1,
    max_weight: int = 10,
    source: Optional[int] = None,
    target: Optional[int] = None,
) -> Graph:
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}.")

    if num_nodes > MAX_NODES:
        raise ValueError(
            f"num_nodes must be <= {MAX_NODES} (node ids are relabelled into a "
            f"{MAX_NODES}-token vocabulary), got {num_nodes}."
        )

    if min_weight <= 0:
        raise ValueError(
            f"min_weight must be > 0 (Dijkstra requires positive weights), "
            f"got {min_weight}."
        )

    if max_weight < min_weight:
        raise ValueError(
            f"max_weight ({max_weight}) must be >= min_weight ({min_weight})."
        )

    max_possible_edges: int = num_nodes * (num_nodes - 1) // 2
    min_spanning_edges: int = num_nodes - 1

    if num_edges is not None:
        if num_edges < 0:
            raise ValueError(f"num_edges must be >= 0, got {num_edges}.")
        if num_edges < min_spanning_edges:
            raise ValueError(
                f"num_edges ({num_edges}) is below the minimum spanning-tree "
                f"size ({min_spanning_edges}) for {num_nodes} nodes.  "
                f"Silently producing more edges than requested would corrupt "
                f"dataset edge-count distributions; pass at least "
                f"{min_spanning_edges}."
            )
        if num_edges > max_possible_edges:
            raise ValueError(
                f"num_edges ({num_edges}) exceeds the maximum possible "
                f"simple-graph edge count ({max_possible_edges}) for "
                f"{num_nodes} nodes."
            )
        target_edge_count: int = num_edges
    elif edge_density is not None:
        if edge_density <= 0 or edge_density > 1:
            raise ValueError(
                f"edge_density must be in (0, 1], got {edge_density}."
            )
        target_edge_count = max(
            min_spanning_edges,
            round(edge_density * max_possible_edges),
        )
        target_edge_count = min(target_edge_count, max_possible_edges)
    else:
        target_edge_count = min_spanning_edges

    if source is not None and not (0 <= source < num_nodes):
        raise ValueError(
            f"source ({source}) is out of range [0, num_nodes)."
        )
    if target is not None and not (0 <= target < num_nodes):
        raise ValueError(
            f"target ({target}) is out of range [0, num_nodes)."
        )
    if source is not None and target is not None and source == target:
        raise ValueError(
            f"source and target must differ, both are {source}."
        )

    rng = random.Random(seed)

    node_order = list(range(num_nodes))
    rng.shuffle(node_order)

    edge_set, edges = _build_spanning_tree(node_order, rng, min_weight, max_weight)

    _add_extra_edges(
        num_nodes,
        edge_set,
        edges,
        target_edge_count,
        rng,
        min_weight,
        max_weight,
    )

    source, target = _resolve_endpoints(num_nodes, source, target, rng)

    # Relabel nodes to a random subset of 0..49
    sampled_ids = sorted(rng.sample(range(MAX_NODES), num_nodes))
    relabel_map = {i: sampled_ids[i] for i in range(num_nodes)}
    
    relabeled_edges = [(relabel_map[u], relabel_map[v], w) for u, v, w in edges]
    relabeled_source = relabel_map[source]
    relabeled_target = relabel_map[target]

    return Graph(
        num_nodes=num_nodes,
        edges=relabeled_edges,
        source=relabeled_source,
        target=relabeled_target,
        seed=seed,
        node_ids=sampled_ids,
    )


if __name__ == "__main__":
    import json

    g = generate_random_connected_graph(
        num_nodes=6,
        seed=42,
        edge_density=0.5,
        min_weight=1,
        max_weight=15,
    )

    print("=== Example Graph ===")
    print(f"Nodes  : {g.num_nodes}")
    print(f"Source : {g.source}  |  Target : {g.target}")
    print(f"Edges  : {len(g.edges)}")

    for u, v, w in g.edges:
        print(f"  {u} -- {v}  (weight={w})")

    print("\nAdjacency list:")

    for node, neighbours in sorted(g.adjacency().items()):
        print(f"  {node}: {neighbours}")

    print("\nto_dict():", json.dumps(g.to_dict(), indent=2))
from __future__ import annotations
from typing import List,Dict
from data.graph_generator import Graph
from data.dijkstra import ShortestPath

_NUM_NODE_TOKENS = 50
_SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[SRC]", "[DST]"]

class GraphTokenizer:
  def __init__(self,min_weight:int=1,max_weight:int=10)->None:
    if min_weight < 1:
      raise ValueError(f"min_weight must be >= 1, got {min_weight}")
    if max_weight <= min_weight:
      raise ValueError(
        f"max_weight must be > min_weight, got max={max_weight}, min={min_weight}"
      )
    self.min_weight = min_weight
    self.max_weight = max_weight
    self.token_to_id : Dict[str, int] = {}
    self.id_to_token : Dict[int, str] = {}
    self._build_vocab()
  
  def _build_vocab(self)->None:
    tokens : List[str] = []
    tokens.extend(_SPECIAL_TOKENS)
    for n in range(_NUM_NODE_TOKENS):
      tokens.append(f"node_{n}")
    for w in range(self.min_weight,self.max_weight+1):
      tokens.append(f"weight_{w}")
    for idx,tok in enumerate(tokens):
      self.token_to_id[tok] = idx
      self.id_to_token[idx] = tok
    self.vocab_size = len(tokens)    
    self.pad_token_id = self.token_to_id["[PAD]"]
    self.bos_token_id = self.token_to_id["[BOS]"]
    self.eos_token_id = self.token_to_id["[EOS]"]
    self.src_token_id = self.token_to_id["[SRC]"]
    self.dst_token_id = self.token_to_id["[DST]"]
    
  def _node_token_id(self,node_id:int)->int:
    if not(0<=node_id < _NUM_NODE_TOKENS):
      raise ValueError(
                f"node_id {node_id} is out of supported range [0, {_NUM_NODE_TOKENS - 1}]"
      )
    return self.token_to_id[f"node_{node_id}"]
  
  def _weight_token_id(self,weight:int)->int:
    if not (self.min_weight <= weight <= self.max_weight):
      raise ValueError(
                f"weight {weight} is outside configured range "
                f"[{self.min_weight}, {self.max_weight}]"
      )
    return self.token_to_id[f"weight_{weight}"]
  
  def _token_id_to_node_id(self,token_id:int)->int:
    tok = self.id_to_token.get(token_id)
    if tok is None or not tok.startswith("node_"):
      raise ValueError(
                f"token id {token_id} ('{tok}') is not a node token"
      )
    return int(tok[len("node_"):])
  
  @staticmethod
  def _canonical_edges(graph:Graph)->List[tuple]:
    canonical : List[tuple] = []
    for (u,v,w) in graph.edges:
      u_c,v_c = (u,v) if u<=v else(v,u)
      canonical.append((u_c, v_c, w))
    canonical.sort()
    return canonical
  def encode_graph(self, graph: Graph) -> List[int]:
    if not graph.edges:
      raise ValueError(
                "encode_graph received a graph with no edges. "
                "All graphs in this project are connected, so a 0-edge graph "
                "indicates a data pipeline error. Empty-adjacency encoding is "
                "unsupported."
      )
    ids : List[int] = []
    # Query block first (positions 0-3)
    ids.append(self.src_token_id)
    ids.append(self._node_token_id(graph.source))
    ids.append(self.dst_token_id)
    ids.append(self._node_token_id(graph.target))
    
    # Edge block follows (positions 4 onward)
    for (u, v, w) in self._canonical_edges(graph):
      ids.append(self._node_token_id(u))
      ids.append(self._node_token_id(v))
      ids.append(self._weight_token_id(w))
    
    return ids
  
  def encode_path(self,shortest_path:ShortestPath)->List[int]:
    ids : List[int] = [self.bos_token_id]
    for node in shortest_path.path:
      ids.append(self._node_token_id(node))
    ids.append(self.eos_token_id)
    return ids
  
  def decode_path(self,token_ids:List[int])->List[int]:
    if not token_ids or token_ids[0] != self.bos_token_id:
      raise ValueError(
                f"decode_path expects sequence starting with BOS "
                f"(id={self.bos_token_id}), got: {token_ids[:5]}"
      )
    path : List[int] = []
    for tid in token_ids[1:]:
      if tid == self.eos_token_id:
        break
      path.append(self._token_id_to_node_id(tid))
    return path

  def __repr__(self) -> str:
    return (
        f"GraphTokenizer("
        f"vocab_size={self.vocab_size}, "
        f"min_weight={self.min_weight}, "
        f"max_weight={self.max_weight})"
    )

  
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import torch

from data.graph_generator import Graph, generate_random_connected_graph, locked_edge_count
from data.dijkstra import run_dijkstra, ShortestPath
from data.tokenizer import GraphTokenizer
from model.model import Transformer
from eval.evaluate import _greedy_decode, _MAX_DECODE_LEN, _decode_token_sequence, _check_path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
_MODEL: Transformer = None
_TOKENIZER: GraphTokenizer = None
_DEVICE: torch.device = None

@app.on_event("startup")
def load_model():
    global _MODEL, _TOKENIZER, _DEVICE
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {_DEVICE}")
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints_run5", "final.pt")
        
    if not os.path.exists(ckpt_path):
        print(f"Warning: Model not found at {ckpt_path}. Endpoint prediction will fail.")
        return
        
    ckpt = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
    model_cfg = ckpt["model_config"]
    tok_kwargs = ckpt["tokenizer_kwargs"]
    
    _TOKENIZER = GraphTokenizer(**tok_kwargs)
    _MODEL = Transformer(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    _MODEL.load_state_dict(ckpt["state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    print("Model loaded successfully!")

class GenerateResponse(BaseModel):
    num_nodes: int
    node_ids: List[int]
    edges: List[List[int]]
    source: int
    target: int
    seed: int

@app.get("/api/generate", response_model=GenerateResponse)
def generate_graph(num_nodes: int = 10, seed: int = None):
    if seed is None:
        import random
        seed = random.randint(0, 1000000)
    
    dynamic_edges = locked_edge_count(num_nodes)
    try:
        graph = generate_random_connected_graph(
            num_nodes=num_nodes,
            seed=seed,
            num_edges=dynamic_edges,
            min_weight=1,
            max_weight=10
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenerateResponse(
        num_nodes=graph.num_nodes,
        node_ids=graph.node_ids,
        edges=[[u, v, w] for u, v, w in graph.edges],
        source=graph.source,
        target=graph.target,
        seed=graph.seed
    )

class PredictRequest(BaseModel):
    num_nodes: int
    node_ids: List[int]
    edges: List[List[int]]
    source: int
    target: int
    seed: int

class PredictResponse(BaseModel):
    dijkstra_path: List[int]
    dijkstra_cost: float
    model_path: List[int]
    model_raw_tokens: List[str]
    valid_path: bool
    is_optimal: bool

@app.post("/api/predict", response_model=PredictResponse)
def predict_path(req: PredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    graph = Graph(
        num_nodes=req.num_nodes,
        node_ids=req.node_ids,
        edges=[tuple(e) for e in req.edges],
        source=req.source,
        target=req.target,
        seed=req.seed
    )
    
    # 1. True path via Dijkstra
    sp = run_dijkstra(graph)
    
    # 2. Predicted path via Transformer
    src_ids = torch.tensor([_TOKENIZER.encode_graph(graph)], dtype=torch.long)
    raw_tokens = _greedy_decode(_MODEL, src_ids, _TOKENIZER, _DEVICE, _MAX_DECODE_LEN)
    
    # Fetch raw token strings for the UI
    raw_token_strs: List[str] = []
    for tid in raw_tokens:
        tok = _TOKENIZER.id_to_token.get(tid, f"<UNK:{tid}>")
        raw_token_strs.append(tok)
        if tok == "<EOS>":
            break
            
    # Use proper harness evaluation functions
    nodes = _decode_token_sequence(raw_tokens, _TOKENIZER)
    if nodes is None:
        nodes = []
        
    valid_path, correct_endpoints, edges_valid, optimal_cost, decoded_cost = _check_path(
        nodes, graph, sp.cost
    )
    
    is_optimal = valid_path and correct_endpoints and optimal_cost
                
    return PredictResponse(
        dijkstra_path=sp.path,
        dijkstra_cost=sp.cost,
        model_path=nodes,
        model_raw_tokens=raw_token_strs,
        valid_path=valid_path and correct_endpoints,
        is_optimal=is_optimal
    )

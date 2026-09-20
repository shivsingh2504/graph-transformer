import os
import sys
import json
import torch
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data.dataset_generator import generate_dataset_split
import api
from eval.evaluate import evaluate_example
from data.graph_generator import Graph
from data.dijkstra import ShortestPath

def test_harness_vs_glue():
    api.load_model()
    
    # We use the current dataset generator. The harness and glue 
    # should agree exactly on all generated graphs, regardless of what 
    # distribution it currently produces.
    eval_split = generate_dataset_split(
        num_examples=1000,
        node_range=(5, 20),
        base_seed=1,
    )
    
    disagreements = 0
    harness_opts = 0
    api_opts = 0
    
    for i, (graph, sp) in enumerate(eval_split.examples):
        # 1. Harness Path
        harness_result = evaluate_example(
            model=api._MODEL,
            graph=graph,
            shortest_path=sp,
            tokenizer=api._TOKENIZER,
            device=api._DEVICE
        )
        
        # 2. API Path
        req = api.PredictRequest(
            num_nodes=graph.num_nodes,
            node_ids=graph.node_ids,
            edges=[list(e) for e in graph.edges],
            source=graph.source,
            target=graph.target,
            seed=graph.seed
        )
        api_result = api.predict_path(req)
        
        harness_opt = harness_result.valid_and_optimal
        api_opt = api_result.is_optimal
        
        if harness_opt:
            harness_opts += 1
        if api_opt:
            api_opts += 1
            
        if harness_opt != api_opt:
            disagreements += 1
            
    assert disagreements == 0, f"Disagreement count: {disagreements}"
    print(f"Harness valid_and_optimal: {harness_opts}/1000")
    print(f"API valid_and_optimal: {api_opts}/1000")
    print(f"Disagreement count: {disagreements}")

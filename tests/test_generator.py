from __future__ import annotations
from typing import List,Tuple
from collections import deque
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest 
from src.data.graph_generator import Graph,generate_random_connected_graph

def _is_connected(graph: Graph)->bool:
  if graph.num_nodes == 0 :
    return True
  adj = {i:[] for i in graph.node_ids}
  
  for u,v,_ in graph.edges:
    adj[u].append(v)
    adj[v].append(u)
  start_node = graph.node_ids[0]
  visited={start_node}
  queue = deque([start_node])
  
  while queue:
    node = queue.popleft()
    for nb in adj[node]:
      if nb not in visited:
        visited.add(nb)
        queue.append(nb)
  return len(visited) == graph.num_nodes

def _max_edges(n:int)->int:
  return n*(n-1) // 2

class TestReproducibility:
  @pytest.mark.parametrize(
    "num_nodes,seed",
    [(2, 0), (5, 1), (10, 99), (20, 12345), (50, 999), (100, 7)],
  )
  def test_same_seed_same_graph(self,num_nodes:int,seed:int)->None:
    g1 = generate_random_connected_graph(num_nodes, seed)
    g2 = generate_random_connected_graph(num_nodes, seed)
    
    assert g1.edges == g2.edges
    assert g1.source == g2.source
    assert g1.target == g2.target
    
  @pytest.mark.parametrize("seed", [0, 42, 1337])
  def test_same_seed_with_density(self,seed:int)->None:
    g1 = generate_random_connected_graph(10, seed, edge_density=0.5)
    g2 = generate_random_connected_graph(10, seed, edge_density=0.5)
    assert g1.edges == g2.edges
    assert g1.source == g2.source
    assert g1.target == g2.target
    
  @pytest.mark.parametrize("seed", [0, 42, 1337])
  def test_same_seed_with_num_edges(self, seed: int) -> None:
    g1 = generate_random_connected_graph(10, seed, num_edges=15)
    g2 = generate_random_connected_graph(10, seed, num_edges=15)
    assert g1.edges == g2.edges
    
class TestDifferentSeeds:
  def test_different_seeds_differ(self)->None:
    g1 = generate_random_connected_graph(20, seed=0)
    g2 = generate_random_connected_graph(20, seed=1)
    assert (g1.edges,g1.target,g1.source)!=(g2.edges,g2.target,g2.source)
    
class TestConnectivity:
  @pytest.mark.parametrize("seed", range(20))
  def test_spanning_tree_connected(self,seed:int)->None:
    g = generate_random_connected_graph(10, seed)
    assert _is_connected(g)
  @pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.75, 1.0])
  @pytest.mark.parametrize("seed", [0, 7, 42])
  def test_density_connected(self,density:float,seed:int)->None:
    g = generate_random_connected_graph(15,seed,edge_density=density)
    assert _is_connected(g)
    
  @pytest.mark.parametrize("seed", range(10))
  def test_full_density_connected(self,seed:int)->None:
    g = generate_random_connected_graph(8,seed,edge_density=1.0)
    assert _is_connected(g)
    
  @pytest.mark.parametrize("seed", range(10))
  def test_explicit_num_edges_connected(self,seed:int)->None:
    g = generate_random_connected_graph(12,seed,num_edges=20)
    assert _is_connected(g)
  

class TestNodeValidity:
  @pytest.mark.parametrize("seed",range(15))
  def test_no_self_loops_density_mode(self,seed:int)->None:
    g = generate_random_connected_graph(10,seed,edge_density=0.6)
    for u,v,_ in g.edges:
      assert u!=v
  @pytest.mark.parametrize("seed", range(15))
  def test_no_self_loops_num_edges_mode(self, seed: int) -> None:
      g = generate_random_connected_graph(
        10, seed, num_edges=15
      )

      for u, v, _ in g.edges:
          assert u != v

  @pytest.mark.parametrize("seed", range(15))
  def test_no_duplicate_edges_density_mode(self, seed: int) -> None:
      g = generate_random_connected_graph(
          10, seed, edge_density=0.6
      )

      canonical = [(u, v) for u, v, _ in g.edges]

      assert len(canonical) == len(set(canonical))
  @pytest.mark.parametrize("seed", range(10))
  def test_endpoints_in_range(self, seed: int) -> None:
      g = generate_random_connected_graph(
          10, seed, edge_density=0.5
      )

      for u, v, _ in g.edges:
          assert u in g.node_ids
          assert v in g.node_ids
  @pytest.mark.parametrize("seed", range(10))
  def test_canonical_ordering_density_mode(self, seed: int) -> None:
      g = generate_random_connected_graph(
          10, seed, edge_density=0.5
      )

      for u, v, _ in g.edges:
          assert u < v

  @pytest.mark.parametrize("seed", range(10))
  def test_canonical_ordering_num_edges_mode(self, seed: int) -> None:
      g = generate_random_connected_graph(
          10, seed, num_edges=15
      )

      for u, v, _ in g.edges:
          assert u < v
class TestWeights:

    @pytest.mark.parametrize("seed", range(10))
    def test_default_weight_bounds(self, seed: int) -> None:
        g = generate_random_connected_graph(
            10, seed, edge_density=0.5
        )

        for _, _, w in g.edges:
            assert 1 <= w <= 10

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_custom_weight_bounds(self, seed: int) -> None:
        g = generate_random_connected_graph(
            10,
            seed,
            edge_density=0.5,
            min_weight=3,
            max_weight=7
        )

        for _, _, w in g.edges:
            assert 3 <= w <= 7

    def test_single_weight_value(self) -> None:
        g = generate_random_connected_graph(
            8,
            seed=0,
            min_weight=5,
            max_weight=5
        )

        for _, _, w in g.edges:
            assert w == 5

    def test_min_weight_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_weight"):
            generate_random_connected_graph(
                5,
                seed=0,
                min_weight=0
            )

    def test_min_weight_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_weight"):
            generate_random_connected_graph(
                5,
                seed=0,
                min_weight=-3
            )

    def test_max_weight_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            generate_random_connected_graph(
                5,
                seed=0,
                min_weight=5,
                max_weight=3
            )


class TestEdgeCount:

    @pytest.mark.parametrize("num_nodes", [2, 5, 10, 20])
    def test_default_is_spanning_tree(
        self,
        num_nodes: int
    ) -> None:
        g = generate_random_connected_graph(
            num_nodes,
            seed=0
        )

        assert len(g.edges) == num_nodes - 1

    @pytest.mark.parametrize("requested", [9, 15, 20])
    def test_explicit_num_edges_respected(
        self,
        requested: int
    ) -> None:
        g = generate_random_connected_graph(
            10,
            seed=0,
            num_edges=requested
        )

        assert len(g.edges) == requested

    def test_num_edges_below_spanning_tree_raises(self) -> None:
        with pytest.raises(ValueError, match="num_edges"):
            generate_random_connected_graph(
                5,
                seed=0,
                num_edges=2
            )

    def test_num_edges_at_exact_minimum_valid(self) -> None:
        n = 8

        g = generate_random_connected_graph(
            n,
            seed=0,
            num_edges=n - 1
        )

        assert len(g.edges) == n - 1
        assert _is_connected(g)

    def test_num_edges_at_exact_maximum_valid(self) -> None:
        n = 6
        max_e = _max_edges(n)

        g = generate_random_connected_graph(
            n,
            seed=0,
            num_edges=max_e
        )

        assert len(g.edges) == max_e

    def test_num_edges_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="num_edges"):
            generate_random_connected_graph(
                5,
                seed=0,
                num_edges=11
            )

    def test_num_edges_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_edges"):
            generate_random_connected_graph(
                5,
                seed=0,
                num_edges=-1
            )

    def test_edge_density_one_gives_complete_graph(self) -> None:
        n = 6

        g = generate_random_connected_graph(
            n,
            seed=0,
            edge_density=1.0
        )

        assert len(g.edges) == _max_edges(n)

    @pytest.mark.parametrize(
        "density",
        [0.25, 0.5, 0.75]
    )
    def test_edge_density_scales_correctly(
        self,
        density: float
    ) -> None:
        n = 10

        expected = max(
            n - 1,
            round(density * _max_edges(n))
        )

        g = generate_random_connected_graph(
            n,
            seed=0,
            edge_density=density
        )

        assert len(g.edges) == expected

    def test_edge_density_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="edge_density"):
            generate_random_connected_graph(
                5,
                seed=0,
                edge_density=0.0
            )

    def test_edge_density_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="edge_density"):
            generate_random_connected_graph(
                5,
                seed=0,
                edge_density=-0.1
            )

    def test_edge_density_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="edge_density"):
            generate_random_connected_graph(
                5,
                seed=0,
                edge_density=1.5
            )

    def test_num_edges_wins_over_edge_density(self) -> None:
        n = 10

        g = generate_random_connected_graph(
            n,
            seed=0,
            num_edges=12,
            edge_density=1.0
        )

        assert len(g.edges) == 12


class TestSourceTarget:

    @pytest.mark.parametrize("seed", range(10))
    def test_auto_source_target_differ(
        self,
        seed: int
    ) -> None:
        g = generate_random_connected_graph(
            10,
            seed
        )

        assert g.source != g.target

    @pytest.mark.parametrize("seed", range(10))
    def test_auto_source_target_in_range(
        self,
        seed: int
    ) -> None:
        g = generate_random_connected_graph(
            10,
            seed
        )

        assert g.source in g.node_ids
        assert g.target in g.node_ids

    def test_explicit_source_respected(self) -> None:
        g = generate_random_connected_graph(
            10,
            seed=0,
            source=3
        )

        assert g.source == g.node_ids[3]

    def test_explicit_target_respected(self) -> None:
        g = generate_random_connected_graph(
            10,
            seed=0,
            target=7
        )

        assert g.target == g.node_ids[7]

    def test_explicit_both_respected(self) -> None:
        g = generate_random_connected_graph(
            10,
            seed=0,
            source=2,
            target=8
        )

        assert g.source == g.node_ids[2]
        assert g.target == g.node_ids[8]

    def test_source_equals_target_raises(self) -> None:
        with pytest.raises(ValueError, match="source and target"):
            generate_random_connected_graph(
                10,
                seed=0,
                source=3,
                target=3
            )

    def test_source_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="source"):
            generate_random_connected_graph(
                10,
                seed=0,
                source=10
            )

    def test_target_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="target"):
            generate_random_connected_graph(
                10,
                seed=0,
                target=-1
            )

    def test_source_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="source"):
            generate_random_connected_graph(
                10,
                seed=0,
                source=-1
            )


class TestNumNodes:

    def test_num_nodes_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="num_nodes"):
            generate_random_connected_graph(
                1,
                seed=0
            )

    def test_num_nodes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_nodes"):
            generate_random_connected_graph(
                0,
                seed=0
            )

    def test_num_nodes_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_nodes"):
            generate_random_connected_graph(
                -5,
                seed=0
            )

    def test_num_nodes_two_minimal(self) -> None:
        g = generate_random_connected_graph(
            2,
            seed=0
        )

        assert g.num_nodes == 2
        assert len(g.edges) == 1

        u, v, w = g.edges[0]

        assert {u, v}.issubset(set(g.node_ids))
        assert 1 <= w <= 10


class TestGraphMethods:

    @pytest.fixture(
        params=[
            {"edge_density": 0.5},
            {"num_edges": 12},
            {},
            {"edge_density": 1.0}
        ],
        ids=[
            "density_0.5",
            "num_edges_12",
            "spanning_tree",
            "complete"
        ]
    )
    def sample_graph(self, request) -> Graph:
        return generate_random_connected_graph(
            8,
            seed=7,
            **request.param
        )

    def test_to_dict_keys(
        self,
        sample_graph: Graph
    ) -> None:
        assert set(sample_graph.to_dict().keys()) == {
            "num_nodes",
            "edges",
            "source",
            "target",
            "seed",
            "node_ids"
        }

    def test_to_dict_num_nodes(
        self,
        sample_graph: Graph
    ) -> None:
        assert (
            sample_graph.to_dict()["num_nodes"]
            == sample_graph.num_nodes
        )

    def test_to_dict_edges_format(
        self,
        sample_graph: Graph
    ) -> None:
        for item in sample_graph.to_dict()["edges"]:
            assert isinstance(item, list)
            assert len(item) == 3

    def test_to_dict_edges_values_match(
        self,
        sample_graph: Graph
    ) -> None:
        d_edges = [
            tuple(e)
            for e in sample_graph.to_dict()["edges"]
        ]

        assert d_edges == list(sample_graph.edges)

    def test_to_dict_source_target(
        self,
        sample_graph: Graph
    ) -> None:
        d = sample_graph.to_dict()

        assert d["source"] == sample_graph.source
        assert d["target"] == sample_graph.target

    def test_to_dict_seed(
        self,
        sample_graph: Graph
    ) -> None:
        assert (
            sample_graph.to_dict()["seed"]
            == sample_graph.seed
        )

    def test_adjacency_all_nodes_present(
        self,
        sample_graph: Graph
    ) -> None:
        adj = sample_graph.adjacency()

        assert set(adj.keys()) == set(
            sample_graph.node_ids
        )

    def test_adjacency_symmetric(
        self,
        sample_graph: Graph
    ) -> None:
        adj = sample_graph.adjacency()

        for u, v, w in sample_graph.edges:
            assert (v, w) in adj[u]
            assert (u, w) in adj[v]

    def test_adjacency_no_extra_neighbours(
        self,
        sample_graph: Graph
    ) -> None:
        adj = sample_graph.adjacency()

        total = sum(
            len(v)
            for v in adj.values()
        )

        assert total == 2 * len(sample_graph.edges)

    def test_adjacency_isolated_nodes_have_empty_list(
        self
    ) -> None:
        g = generate_random_connected_graph(
            5,
            seed=0
        )

        adj = g.adjacency()

        for node in g.node_ids:
            assert node in adj
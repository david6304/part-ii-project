import pytest
import pandas as pd
from src.utils import graph_utils
from src.utils.graph_utils import remove_bidirectional_edges
import networkx as nx

@pytest.fixture
def example_graph():
    return pd.DataFrame(
        [[0,1,0],[0,0,1],[0,0,0]],
        columns=['A','B','C'],
        index=['A','B','C']
    )

def test_topological_sort(example_graph):
    order = graph_utils.topological_sort(example_graph)
    assert order == ['A', 'B', 'C']
    
def test_bidir_removal():
    """Test bidirectional edge removal"""
    # Create cyclic graph: A <-> B
    cyclic_adj = pd.DataFrame(
        [[0, 1], [1, 0]],
        columns=['A', 'B'],
        index=['A', 'B']
    )
    
    # Remove bidirectional edges
    acyclic_adj = remove_bidirectional_edges(cyclic_adj)
    
    # Check if acyclic
    G = nx.DiGraph(acyclic_adj.values)
    assert not list(nx.simple_cycles(G)), "Graph still contains cycles"

def test_priority_removal():
    """Test priority-based removal"""
    cyclic_adj = pd.DataFrame(
        [[0, 1], [1, 0]],
        columns=['A', 'B'],
        index=['A', 'B']
    )
    
    acyclic_adj = remove_bidirectional_edges(
        cyclic_adj,
        strategy="priority",
        priority_edges={('A', 'B')}
    )
    
    assert acyclic_adj.loc['A', 'B'] == 1
    assert acyclic_adj.loc['B', 'A'] == 0
import networkx as nx
import pandas as pd
from typing import List
import random
import pandas as pd
from typing import Optional, Set, Tuple
import bnlearn as bn
from typing import List
import pickle as pkl
import os

def topological_sort(adj_matrix: pd.DataFrame) -> List[str]:
    """Get topological order of nodes from adjacency matrix"""
    G = nx.DiGraph(adj_matrix)
    return list(nx.topological_sort(G))

def remove_bidirectional_edges(
    graph: pd.DataFrame,
    strategy: str = "random",
    edge_strength: Optional[pd.DataFrame] = None,
    priority_edges: Optional[Set[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Remove one edge from each bidirectional pair using specified strategy.
    
    Args:
        graph: Adjacency matrix (binary or weighted)
        strategy: Removal method - "random", "strength", or "priority"
        edge_strength: Matrix of edge strengths (required for "strength" strategy)
        priority_edges: Set of (src, tgt) edges to preserve (for "priority" strategy)
    
    Returns:
        Acyclic adjacency matrix
    """
    graph = graph.copy()
    bidir_pairs = []
    
    # Find all bidirectional edges
    for src in graph.columns:
        for tgt in graph.index:
            if graph.loc[src, tgt] != 0 and graph.loc[tgt, src] != 0:
                bidir_pairs.append((src, tgt))
    
    # Process each bidirectional pair
    for src, tgt in bidir_pairs:
        if strategy == "random":
            if random.random() < 0.5:
                graph.loc[src, tgt] = 0
            else:
                graph.loc[tgt, src] = 0
                
        elif strategy == "strength":
            if edge_strength is None:
                raise ValueError("edge_strength required for 'strength' strategy")
            if edge_strength.loc[src, tgt] >= edge_strength.loc[tgt, src]:
                graph.loc[tgt, src] = 0
            else:
                graph.loc[src, tgt] = 0
                
        elif strategy == "priority":
            if priority_edges is None:
                priority_edges = set()
            if (src, tgt) in priority_edges:
                graph.loc[tgt, src] = 0
            elif (tgt, src) in priority_edges:
                graph.loc[src, tgt] = 0
            else:
                if random.random() < 0.5:
                    graph.loc[src, tgt] = 0
                else:
                    graph.loc[tgt, src] = 0
                    
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return graph

def get_dag(dataset_name: str) -> pd.DataFrame:
    """
    Retrieves the DAG for a specified Bayesian network dataset from bnlearn.

    Parameters:
    - dataset_name (str): The name of the Bayesian network dataset (e.g., 'asia').

    Returns:
    - Tuple: A tuple containing the model and adjacency matrix of the DAG.
                       
    Raises:
    - ValueError: If the specified dataset is not found in bnlearn's available DAGs.
    """
    # Import the DAG using bnlearn
    try:
        file_path = f"src/ground_truth_graphs/{dataset_name}.pkl"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                dag = pkl.load(f)
                adj_mat = dag["adjmat"]
                model = dag["model"]
        else:
            dag = bn.import_DAG(dataset_name)
            adj_mat = dag["adjmat"]
            model = dag["model"]
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pkl.dump(dag, f)
                
    except Exception as e:
        raise ValueError(f"Failed to import DAG '{dataset_name}'. Ensure the dataset name is correct.") from e
    
    return model, adj_mat.astype(int)

if __name__ == "__main__":
    # Test topological sort
    adj = pd.DataFrame(
        [[0,1],[0,0]],
        columns=['A','B'],
        index=['A','B']
    )
    print("Topological order:", topological_sort(adj))  # Should be ['A', 'B']
    
    # Test get_adjacency_matrix
    model, adj_mat = get_dag("asia")
    
    print("Asia DAG adjacency matrix:")
    print(adj_mat)
    
    print("Asia CPDs:")
    print(model)
    
import pandas as pd
from causallearn.search.PermutationBased.GRaSP import grasp

def run_grasp(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run GRaSP algorithm for causal discovery
    
    Args:
        data: Input dataframe with variables as columns
    
    Returns:
        Adjacency matrix as DataFrame (1 = edge exists, 0 = no edge)
    """
    # Convert to numpy array
    numpy_data = data.values.astype(float)
    
    # Run GRaSP
    G = grasp(numpy_data)

    # Convert to adjacency matrix with correct format
    adj_matrix = [[1 if G.graph[i,j] == -1 else 0 for j in range(len(G.graph[0]))] for i in range(len(G.graph))]
    
    # Convert to DataFrame with original column names
    return pd.DataFrame(
        adj_matrix,
        columns=data.columns,
        index=data.columns
    )

def threshold_grasp_matrix(
    grasp_matrix: pd.DataFrame,
    edge_threshold: float = 0.5
) -> pd.DataFrame:
    """Convert weighted adjacency matrix to binary"""
    return (grasp_matrix.abs() > edge_threshold).astype(int)

if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    
    df = load_synthetic_data()
    grasp_adj = run_grasp(df)
    print("GRaSP adjacency matrix:\n", grasp_adj)
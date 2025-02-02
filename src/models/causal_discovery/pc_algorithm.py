import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

def run_pc_algorithm(
    data: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str = "chisq",
    verbose: bool = False
) -> pd.DataFrame:
    """Run PC algorithm to learn causal graph"""
    # Convert to numpy array
    numpy_data = data.values.astype(float)
    
    # Run PC
    cg = pc(
        data=numpy_data,
        alpha=alpha,
        indep_test=indep_test,
        verbose=verbose,
        show_progress=False
    )
    
    # Convert to adjacency matrix
    adj_matrix = cg.G.graph.T.astype(int)
    return pd.DataFrame(
        adj_matrix,
        columns=data.columns,
        index=data.columns
    )

if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    df = load_synthetic_data()
    learned_graph = run_pc_algorithm(df)
    print("Learned adjacency matrix:\n", learned_graph)
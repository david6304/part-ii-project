import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def plot_adjacency_comparison(
    true_adj: pd.DataFrame,
    learned_adj: pd.DataFrame,
    title: str = "Graph Comparison"
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Ground truth graph
    G_true = nx.DiGraph(true_adj.values)
    nx.draw(G_true, ax=ax1, with_labels=True, node_color='lightgreen')
    ax1.set_title("Ground Truth Graph")
    
    # Learned graph
    G_learned = nx.DiGraph(learned_adj.values)
    nx.draw(G_learned, ax=ax2, with_labels=True, node_color='lightblue')
    ax2.set_title("Learned Graph")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
import pytest
import torch
import pandas as pd
from src.models.causal_gnns import gcn, trainer

@pytest.fixture
def synthetic_adj():
    return torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=torch.float32)

def test_gnn_forward_pass(synthetic_adj):
    model = gcn.CausalGNN(node_dim=1, adj_matrix=synthetic_adj)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = model(x)
    assert out.shape == (2, 3)
    
def test_gnn_training(synthetic_adj):
    # Create circular data matching the adjacency
    data = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    # Should learn to replicate inputs through the graph
    gnn_trainer = trainer.GNNTrainer(
        gcn.CausalGNN(node_dim=1, adj_matrix=synthetic_adj)
    )
    
    # Test single batch training
    batch = data[:2]
    loss = gnn_trainer.train_epoch([batch])
    assert isinstance(loss, float)
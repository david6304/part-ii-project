import pandas as pd
from src.models.causal_discovery import grasp

def test_grasp_discovery():
    """Test GRaSP on synthetic data"""
    data = pd.DataFrame({
        'A': [0,0,1,1],
        'B': [0,1,1,1],
        'C': [0,1,1,1]
    })
    
    adj = grasp.run_grasp(data)
    assert adj.shape == (3, 3)
    assert (adj.values >= 0).all() and (adj.values <= 1).all()
    
    # Test thresholding
    thresh_adj = grasp.threshold_grasp_matrix(adj)
    assert thresh_adj.isin([0,1]).all().all()
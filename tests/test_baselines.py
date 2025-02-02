import pytest
import pandas as pd
import numpy as np
from src.models.baselines import xgb, mlp, hierarchical_xgb
from src.evaluation import metrics

@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        'A': [0, 1, 0, 1],
        'B': [1, 0, 1, 0],
        'target': [0, 0, 1, 1]
    })
    return df, df.copy()

def test_xgb_baseline(dummy_data):
    train, val = dummy_data
    model = xgb.XGBBaseline(target_node="target", task="classification", num_classes=2)
    results = model.train(train, val)
    assert 0 <= results['val_acc'] <= 1

def test_mlp_baseline(dummy_data):
    train, val = dummy_data
    model = mlp.MLPBaseline("target", task="classification", num_classes=2)
    results = model.train(train, val, epochs=5)
    assert 0 <= results['val_acc'] <= 1

def test_hierarchical_xgb():
    """Test hierarchical model with simple graph"""
    # A -> B -> C
    adj = pd.DataFrame(
        [[0,1,0],[0,0,1],[0,0,0]],
        columns=['A','B','C'],
        index=['A','B','C']
    )
    
    # Synthetic data
    train = pd.DataFrame({
        'A': [0,0,1,1],
        'B': [0,1,1,1],
        'C': [0,1,1,1]
    })
    val = train.copy()
    
    model = hierarchical_xgb.HierarchicalXGB(adj)
    results = model.train(train, val)
    
    # Verify execution order
    assert model.execution_order == ['B', 'C']
    
    # Verify basic functionality
    assert 'B' in results
    assert 0.5 <= results['B']['val_acc'] <= 1.0

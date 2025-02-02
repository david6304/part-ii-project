import pandas as pd
from src.evaluation import metrics

def test_accuracy_metrics():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    results = metrics.calculate_accuracy(y_true, y_pred)
    assert results['accuracy'] == 0.75
    assert results['n_correct'] == 3
    
def test_shd_calculation():
    true_adj = pd.DataFrame([[0,1],[0,0]], columns=['A','B'], index=['A','B'])
    learned_adj = pd.DataFrame([[0,0],[1,0]], columns=['A','B'], index=['A','B'])
    shd = metrics.structural_hamming_distance(true_adj, learned_adj)
    assert shd["directed_shd"] == 2
    assert shd["extra_edges"] == 0
    assert shd["missing_edges"] == 0
    
def test_ace_calculation():
    """Test ACE calculation on synthetic data"""
    # Ground truth: Smoking -> Lung Cancer
    data = pd.DataFrame({
        'smoke': [0,0,1,1],
        'lung': [0,0,1,1]  # Perfect correlation
    })
    graph = pd.DataFrame(
        [[0,1],[0,0]],
        columns=['smoke','lung'],
        index=['smoke','lung']
    )
    
    ace = metrics.calculate_ace(
        data=data,
        treatment='smoke',
        outcome='lung',
        graph=graph
    )
    assert abs(ace - 1.0) < 0.01  # Perfect causal effect
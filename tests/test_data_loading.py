import pytest
from pathlib import Path
from src.data_processing import synthetic_data, data_loader


def test_data_generation():
    """Verify Asia dataset generation"""
    df = synthetic_data.generate_asia_data(save=False)
    assert len(df) == 1000
    assert set(df.columns) == {'smoke', 'tub', 'lung', 'bronc', 'asia', 'xray', 'dysp', 'either'}

def test_data_loading():
    """Test loading from disk"""
    df = data_loader.load_synthetic_data()
    assert Path("data/synthetic/asia/base/asia_base.csv").exists()
    assert not df.empty
    assert df.isna().sum().sum() == 0
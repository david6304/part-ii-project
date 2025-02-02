from pathlib import Path
import pandas as pd
from src.utils.config_loader import load_config

def load_synthetic_data(dataset: str = "asia", variant: str = "base") -> pd.DataFrame:
    """Load synthetic dataset from CSV"""
    config = load_config()
    try:
        path = config['data']['synthetic'][dataset] / f"{dataset}_{variant}.csv"
    except KeyError:
        raise ValueError(f"Unsupported dataset: {dataset}")

    
    return pd.read_csv(path)

# Test
if __name__ == "__main__":
    df = load_synthetic_data()
    print("Loaded data shape:", df.shape)
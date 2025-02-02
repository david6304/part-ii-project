import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 0
) -> tuple:
    """Split data into train/val/test and normalize"""
    # Split
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    return train, val, test

if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    df = load_synthetic_data()
    train, val, test = preprocess_data(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
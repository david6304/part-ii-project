import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# Configuration for synthetic datasets
SYNTHETIC_DATASETS = {
    "asia": {
        "path": "data/synthetic/asia/",
        "target_col": "dysp",  # Example target variable
        "task": "classification"
    },
    "alarm": {
        "path": "data/synthetic/alarm/",
        "target_col": "BP",
        "task": "classification"
    },
    "auto_mpg": {
        "path": "data/synthetic/auto_mpg/",
        "target_col": "mpg",
        "task": "regression"
    }
}

def load_synthetic_variations(dataset_name, variations_to_include=None):
    """
    Load specified variations and files for a given synthetic dataset.

    Args:
        dataset_name (str): Name of the synthetic dataset (e.g., 'asia').
        variations_to_include (list, optional): List of variations to include (e.g., ['base', 'flipped']).
                                                If None, all variations are included.
    Returns:
        List of tuples: (data, target_col, task, variation, file_name).
    """
    # Ensure the dataset is configured
    if dataset_name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Dataset {dataset_name} is not configured in SYNTHETIC_DATASETS.")

    # Get dataset configuration
    config = SYNTHETIC_DATASETS[dataset_name]
    dataset_path = config["path"]
    target_col = config["target_col"]
    task = config["task"]

    # Prepare to load the specified variations
    variations = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # Traverse variations (subdirectories) and files
    for variation in os.listdir(dataset_path):
        variation_path = os.path.join(dataset_path, variation)
        if os.path.isdir(variation_path):  # Only process directories
            if variations_to_include is None or variation in variations_to_include:
                for file_name in os.listdir(variation_path):
                    if file_name.endswith(".csv"):
                        file_path = os.path.join(variation_path, file_name)
                        data = pd.read_csv(file_path)
                        variations.append((data, target_col, task, variation, file_name))

    return variations

def split_data(data, target_col, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target_col (str): The name of the target column.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def calculate_metrics(y_true, y_pred, task):
    """
    Calculate evaluation metrics based on the task type.
    
    Args:
        y_true (array-like): True labels or values.
        y_pred (array-like): Predicted labels or values.
        task (str): Task type ('classification' or 'regression').

    Returns:
        dict: Metrics dictionary (Accuracy for classification, MSE for regression).
    """
    if task == "classification":
        accuracy = accuracy_score(y_true, y_pred)
        return {"Accuracy": accuracy}
    elif task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return {"MSE": mse}
    else:
        raise ValueError(f"Unknown task type: {task}. Must be 'classification' or 'regression'.")
import xgboost as xgb
from misc.utils import split_data, calculate_metrics

def train_and_evaluate_xgboost(data, target_col, task, variation, file_name):
    """
    Train and evaluate an XGBoost model on a given dataset.
    
    Args:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target_col (str): The name of the target column.
        task (str): The type of task ('classification' or 'regression').
        variation (str): The dataset variation (e.g., 'base', 'flipped').
        file_name (str): The name of the dataset file.

    Returns:
        dict: A dictionary containing evaluation metrics and metadata.
    """
    # Split data into features and target
    X_train, X_test, y_train, y_test = split_data(data, target_col)

    # Set up model parameters
    if task == "classification":
        model = xgb.XGBClassifier(random_state=0)
    elif task == "regression":
        model = xgb.XGBRegressor(random_state=0)
    else:
        raise ValueError(f"Unknown task type: {task}. Must be 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions, task)
    metrics.update({"Model": "XGBoost", "Variation": variation, "File": file_name})

    return metrics
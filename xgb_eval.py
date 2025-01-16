import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Dataset paths
DATASETS = {
    "asia": "data/synthetic/asia/",
    "alarm": "data/synthetic/alarm/",
    "auto_mpg": "data/synthetic/auto_mpg/"
}

VARIATIONS = ["base", "flipped", "gaussian", "missing"]
SAMPLE_SIZES = [1000, 5000, 10000]

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=["Dataset", "Variation", "File", "Metric", "Value"])

# Will use later when I know which datasets I want to optimize for
def optimize_xgboost(X_train, y_train, task="classification"):
    # Define parameter grid
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    if task == "classification":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        model = xgb.XGBRegressor()

    # Grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy" if task == "classification" else "neg_mean_squared_error", cv=3, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    return grid_search.best_estimator_

def train_and_evaluate_xgboost(dataset, target_col, task="classification"):
    global results
    dataset = dataset.dropna(subset=[target_col])
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if task == "classification":
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Store metrics in results
        new_results = pd.DataFrame([
            {"Dataset": current_dataset, "Variation": current_variation, "File": current_file, "Metric": "Accuracy", "Value": accuracy_score(y_test, y_pred)},
            {"Dataset": current_dataset, "Variation": current_variation, "File": current_file, "Metric": "F1 Score", "Value": f1_score(y_test, y_pred, average="weighted")}
        ])
    elif task == "regression":
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Store metrics in results
        new_results = pd.DataFrame([
            {"Dataset": current_dataset, "Variation": current_variation, "File": current_file, "Metric": "MAE", "Value": mean_absolute_error(y_test, y_pred)},
            {"Dataset": current_dataset, "Variation": current_variation, "File": current_file, "Metric": "MSE", "Value": mean_squared_error(y_test, y_pred)}
        ])

    # Concatenate new results to the global results DataFrame
    results = pd.concat([results, new_results], ignore_index=True)
# Iterate over datasets and variations
for dataset_name, base_path in DATASETS.items():
    for variation in VARIATIONS:
        variation_path = os.path.join(base_path, variation)
        
        if os.path.exists(variation_path):
            # Iterate over all CSV files in the current directory
            for file_name in os.listdir(variation_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(variation_path, file_name)
                    print(f"Processing {file_path}")
                    
                    # Load the dataset
                    data = pd.read_csv(file_path)

                    # Define target column and task type
                    if dataset_name == "auto_mpg":
                        target = "mpg"  # Regression task
                        task_type = "regression"
                    else:
                        target = "dysp" if dataset_name == "asia" else "HR"
                        task_type = "classification"

                    # Track metadata
                    current_dataset = dataset_name
                    current_variation = variation
                    current_file = file_name

                    # Train and evaluate the model
                    print(f"Evaluating on file: {file_name}")
                    train_and_evaluate_xgboost(data, target_col=target, task=task_type)

results.to_csv("xgboost_results.csv", index=False)
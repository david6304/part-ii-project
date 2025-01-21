import pandas as pd
from models.xgboost_eval import train_and_evaluate_xgboost
from models.gnn_eval import train_and_evaluate_gnn
from models.dnn_eval import train_and_evaluate_dnn
# from models.bayesian_eval import train_and_evaluate_bayesian
# from models.cem_eval import train_and_evaluate_cem
from models.utils import load_synthetic_variations

# Define models and their evaluation functions
MODELS = {
    "XGBoost": train_and_evaluate_xgboost,
    "GNN": train_and_evaluate_gnn,
    "DNN": train_and_evaluate_dnn,
    # "Bayesian Network": train_and_evaluate_bayesian,  # Add evaluation function here
    # "Causal GNN": train_and_evaluate_cem,  # Add evaluation function here
}

def evaluate_synthetic_datasets():
    """
    Evaluate all synthetic datasets and variations.
    Returns:
        results: List of dictionaries containing evaluation results.
    """
    synthetic_datasets = [
        "asia",
        # "alarm",
        # "auto_mpg",
        ] 
    results = []

    for dataset_name in synthetic_datasets:
        data_files = load_synthetic_variations(dataset_name, ["base"])
        for data, target_col, task, variation, file_name in data_files:
            print(f"Evaluating {dataset_name} ({variation}) with {len(data)} samples...")
            for model_name, eval_func in MODELS.items():
                print(f"Running {model_name}...")
                result = eval_func(data, target_col, task, variation, file_name)
                result["Dataset"] = dataset_name
                result["Variation"] = variation
                result["File"] = file_name
                result["Model"] = model_name
                results.append(result)

    return results

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_synthetic_datasets()

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/synthetic_results.csv", index=False)
    print("Synthetic dataset evaluation completed. Results saved to results/synthetic_results.csv.")
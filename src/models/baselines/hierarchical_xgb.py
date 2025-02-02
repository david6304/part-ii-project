from typing import Dict, List, Optional
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
from src.utils.graph_utils import topological_sort


def are_floats_whole_numbers(labels: pd.Series) -> bool:
    """
    Checks if all float labels in the series are whole numbers.

    Args:
        labels (pd.Series): Series containing label values.

    Returns:
        bool: True if all float labels are whole numbers, False otherwise.
    """
    float_labels = labels[labels.apply(lambda x: isinstance(x, float))]
    return float_labels.apply(float.is_integer).all()


class HierarchicalXGB:
    def __init__(
        self,
        graph: pd.DataFrame,
        default_params: Optional[Dict] = None,
        target_node: str = None,
    ):
        """
        Initializes the HierarchicalXGB model with dynamic task inference.

        Args:
            graph (pd.DataFrame): Adjacency matrix representing the graph structure.
            default_params (Dict, optional): Default parameters for XGBoost models. 
                Specific node parameters can be included during training. Defaults to None.
        """
        self.graph = graph
        self.models = {}
        self.root_nodes = set(graph.columns[graph.sum() == 0])
        self.execution_order = [node for node in topological_sort(graph) if node not in self.root_nodes]
        self.default_params = default_params if default_params else {}
        self.target_node = target_node

    def _infer_task_type(self, y: pd.Series) -> Dict:
        """
        Infers the task type for a given target column.

        Args:
            y (pd.Series): Target column.

        Returns:
            Dict: Dictionary containing 'task' and 'num_classes' (if classification).
        """
        dtype = y.dtype

        # Check if labels are integers
        if pd.api.types.is_integer_dtype(y):
            
            unique_values = y.dropna().unique()
            num_unique = len(unique_values)
            
            if num_unique == 2:
                return {"task": "classification", "num_classes": 2}
            else:
                return {"task": "classification", "num_classes": num_unique}
            
        elif pd.api.types.is_float_dtype(y):
            # Check if all float labels are whole numbers
            if are_floats_whole_numbers(y):
                unique_ints = y.astype(int).dropna().unique()
                num_unique_ints = len(unique_ints)
                if num_unique_ints == 2:
                    return {"task": "classification", "num_classes": 2}
                else:
                    return {"task": "classification", "num_classes": num_unique_ints}
            else:
                return {"task": "regression"}
        else:
            raise ValueError(
                f"Unsupported data type for target column: {dtype}. Must be integer or float."
            )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Trains the HierarchicalXGB model with dynamic task inference.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            params (Dict, optional): Additional parameters for XGBoost models. 
                Node-specific parameters can be included as nested dictionaries. Defaults to None.

        Returns:
            Dict: Dictionary containing training and validation metrics for each node.
        """
        if params is None:
            params = {}

        results = {}
        for node in self.execution_order:
            
            # Infer task type
            task_info = self._infer_task_type(train_data[node])
            task = task_info["task"]
            num_classes = task_info.get("num_classes", None)

            # Identify parent nodes
            parents = list(self.graph.loc[:, node][self.graph.loc[:, node] == 1].index)

            print(
                f"Training {node} | Task: {task} | Num Classes: {num_classes} | Parents: {parents}"
            )

            # Prepare data
            X_train = train_data[parents]
            X_val = val_data[parents]
            y_train = train_data[node]
            y_val = val_data[node]

            # Handle label encoding based on task
            if task == "classification":
                y_train = y_train.astype(int)
                y_val = y_val.astype(int)
            elif task == "regression":
                y_train = y_train.astype(float)
                y_val = y_val.astype(float)

            # Initialize and train the appropriate model
            if task == "classification":
                if num_classes == 2:
                    model = XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **self.default_params,
                        **params.get(node, {}),
                    )
                else:
                    model = XGBClassifier(
                        objective="multi:softprob",
                        num_class=num_classes,
                        eval_metric="mlogloss",
                        **self.default_params,
                        **params.get(node, {}),
                    )
            elif task == "regression":
                model = XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_estimators=100,
                    **self.default_params,
                    **params.get(node, {}),
                )

            # Fit the model
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            self.models[node] = (model, parents)

            # Evaluate the model
            if task == "classification":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                train_acc = accuracy_score(y_train, train_preds)
                val_acc = accuracy_score(y_val, val_preds)

                if num_classes == 2:
                    train_f1 = f1_score(y_train, train_preds)
                    val_f1 = f1_score(y_val, val_preds)
                else:
                    train_f1 = f1_score(
                        y_train, train_preds, average="weighted"
                    )
                    val_f1 = f1_score(y_val, val_preds, average="weighted")

                results[node] = {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "train_f1": train_f1,
                    "val_f1": val_f1,
                }
            elif task == "regression":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                train_mse = mean_squared_error(y_train, train_preds)
                val_mse = mean_squared_error(y_val, val_preds)
                results[node] = {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                }

        return results

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions for the input data in causal order.

        Args:
            X (pd.DataFrame): Input dataset.

        Returns:
            dict: Predictions for target node in the graph.
        """
        preds = {}
        for node in self.execution_order:
            model, parents = self.models[node]
            preds[node] = model.predict(X[parents])
                
        return preds[self.target_node] if self.target_node else preds
    
if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    from src.data_processing.preprocessor import preprocess_data
    from src.utils.graph_utils import get_adjacency_matrix 
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load and preprocess data
    dataset = "alarm"
    df = load_synthetic_data(dataset=dataset)
    train, val, test = preprocess_data(df)
    target_node = "BP"

    # Retrieve the adjacency matrix as a DataFrame
    adj_matrix_df = get_adjacency_matrix(dataset_name=dataset)
    
    # Initialize and train the HierarchicalXGB model
    model = HierarchicalXGB(graph=adj_matrix_df, target_node=target_node)
    results = model.train(train, val)
    
    # Evaluate the model on the test set
    test_preds = model.predict(test)
    test_metrics = {}

    for node in model.execution_order:
        true_labels = test[node]
        pred_labels = test_preds[node]
        
        test_metrics[node] = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels, average="weighted"),
            "recall": recall_score(true_labels, pred_labels, average="weighted"),
            "f1": f1_score(true_labels, pred_labels, average="weighted"),
        }
    
    print(f"Test Metrics for Hierarchical XGB predicting {target_node} in {dataset}:")
    print(test_metrics[target_node])
     
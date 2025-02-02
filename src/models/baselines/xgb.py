import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple

class XGBBaseline:
    def __init__(self, target_node: str, task: str = "classification", num_classes=None):
        self.target = target_node
        
        if task == "classification":
            if num_classes is None or num_classes == 2:
                # Default to binary
                objective = "binary:logistic"
                eval_metric = "logloss"
            else:
                objective = "multi:softmax"
                eval_metric = "mlogloss"
            self.model = xgb.XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                n_estimators=100,
                early_stopping_rounds=10
                )
        elif task == "regression":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                n_estimators=100,
                early_stopping_rounds=10
            )  
        else:
            raise ValueError(f"Unsupported task: {task} - use 'classification' or 'regression'")
        

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame
    ) -> Dict:
        """Train model on one target node"""
        X_train = train_data.drop(columns=[self.target])
        y_train = train_data[self.target]
        X_val = val_data.drop(columns=[self.target])
        y_val = val_data[self.target]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        return {
            # "train_acc": accuracy_score(y_train, self.model.predict(X_train)),
            # "val_acc": accuracy_score(y_val, self.model.predict(X_val))
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X.drop(columns=[self.target]))

# Example usage
if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    from src.data_processing.preprocessor import preprocess_data
    
    dataset = "alarm"
    df = load_synthetic_data(dataset=dataset)
    train, val, test = preprocess_data(df)
    target_node = "BP"
    num_classes = df[target_node].nunique()
    
    # Test prediction on 'dysp' node
    model = XGBBaseline(target_node=target_node, task="classification", num_classes=num_classes)
    metrics = model.train(train, val)
    print(f"Train Acc: {metrics['train_acc']:.3f}, Val Acc: {metrics['val_acc']:.3f}")
    
    # Predict on test set
    test_preds = model.predict(test)
    test_acc = accuracy_score(test[target_node], test_preds)
    print(f"Test Metrics for XGB predicting {target_node} in {dataset}:")
    print(f"Test Acc: {test_acc:.3f}")
    # test_f1 = f1_score(test[target_node], test_preds)
    # print(f"Test F1: {test_f1:.3f}")
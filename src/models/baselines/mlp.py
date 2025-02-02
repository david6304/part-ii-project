import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from typing import Dict, List

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, output_dim: int):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPBaseline:
    def __init__(
        self, 
        target_node: str, 
        hidden_dims: List[int] = [64, 32], 
        dropout: float = 0.1, 
        task: str = "classification",
        num_classes: int = None
    ):
        self.target_node = target_node
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.task = task
        self.num_classes = num_classes
        self.model = None
        self.best_model = None
        self.input_dim = None
    
    def get_loss_function(self):
        if self.task == "classification":
            if self.num_classes is None or self.num_classes == 2:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task: {self.task} - use 'classification' or 'regression'")
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        learning_rate: float = 0.001,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32
    ) -> Dict:
        # Prepare features and labels
        X_train = train_data.drop(columns=[self.target_node]).values.astype(np.float32)
        y_train = train_data[self.target_node].values
        X_val = val_data.drop(columns=[self.target_node]).values.astype(np.float32)
        y_val = val_data[self.target_node].values
        
        # Encode labels if classification
        if self.task == "classification":
            if self.num_classes == 2:
                # Binary labels: Ensure labels are float for BCEWithLogitsLoss
                y_train = y_train.astype(np.float32)
                y_val = y_val.astype(np.float32)
            elif self.num_classes > 2:
                # Multi-class labels: Ensure labels are integers for CrossEntropyLoss
                y_train = y_train.astype(int)
                y_val = y_val.astype(int)
        
        self.input_dim = X_train.shape[1]
        output_dim = 1 if self.task == "regression" or (self.task == "classification" and self.num_classes == 2) else self.num_classes
        
        if self.model is None:
            self.model = MLP(self.input_dim, self.hidden_dims, self.dropout, output_dim)
        
        criterion = self.get_loss_function()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create TensorDatasets and DataLoaders
        if self.task == "classification" and self.num_classes > 2:
            # For multi-class, labels should be LongTensor
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)
            )
        else:
            # For binary classification and regression
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32)
            )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                self.best_model = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Stopping early at epoch {epoch+1}")
                    break
        
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        # Calculate accuracy
        train_preds = self.predict(train_data)
        val_preds = self.predict(val_data)
        
        if self.task == "classification":
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
        else:
            # Define regression metrics if needed
            train_acc = None
            val_acc = None
        
        return {"train_acc": train_acc, "val_acc": val_acc}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_tensor = torch.tensor(X.drop(columns=[self.target_node]).values.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            if self.task == "classification":
                if self.num_classes == 2:
                    preds = (torch.sigmoid(outputs) >= 0.5).float().numpy()
                else:
                    preds = torch.argmax(outputs, dim=1).numpy()
            elif self.task == "regression":
                preds = outputs.numpy()
            return preds
    
# Example usage
if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    from src.data_processing.preprocessor import preprocess_data
    
    dataset = "alarm"
    df = load_synthetic_data(dataset=dataset)
    train, val, test = preprocess_data(df)
    target_node = "BP"
    num_classes = df[target_node].nunique()
    
    # Example for MLP
    mlp_model = MLPBaseline(target_node=target_node, task="classification", num_classes=num_classes)
    mlp_metrics = mlp_model.train(train, val, epochs=100, patience=20)
    mlp_test_preds = mlp_model.predict(test)
    mlp_test_acc = accuracy_score(test[target_node], mlp_test_preds)
    print(f"Test Metrics for MLP predicting {target_node} in {dataset}:")
    print(f"Train Accuracy: {mlp_metrics['train_acc']:.3f}")
    print(f"Val Accuracy: {mlp_metrics['val_acc']:.3f}")
    print(f"Test Accuracy: {mlp_test_acc:.3f}")
    
    # mlp_test_f1 = f1_score(test[target_node], mlp_test_preds)
    # print(f"MLP Test F1: {mlp_test_f1:.3f}")

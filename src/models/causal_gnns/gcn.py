import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score


class GCN(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear(x)
        return x


class GCNBaseline:
    def __init__(
        self, 
        target_node: str, 
        adj_mat: pd.DataFrame, 
        hidden_dim: int = 64,
        task: str = "classification",
        num_classes: Optional[int] = None
    ):
        """
        Initializes the GCNBaseline model.

        Args:
            target_node (str): The name of the target node.
            adj_mat (pd.DataFrame): Adjacency matrix as a DataFrame.
            hidden_dim (int, optional): Number of hidden units. Defaults to 64.
            task (str, optional): Task type: 'classification' or 'regression'. Defaults to 'classification'.
            num_classes (int, optional): Number of classes for classification. Defaults to None (binary classification).
        """
        self.target_node = target_node
        self.adj_mat = adj_mat
        self.hidden_dim = hidden_dim
        self.task = task
        self.num_classes = num_classes
        self.model = None
        self.best_model = None
        self.edge_index = None
        self.target_idx = self.adj_mat.columns.get_loc(self.target_node)
    
    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        edge_indices = []

        # Iterate over the adjacency matrix rows to create edge indices
        for src_idx, row in enumerate(self.adj_mat.values):
            for tgt_idx, val in enumerate(row):
                if val != 0:
                    edge_indices.append([src_idx, tgt_idx])

        # Convert edge indices to a tensor
        self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        data_list = []
        for _, row in df.iterrows():
            x = torch.tensor(row.values.astype(np.float32)).view(-1, 1)
            x[self.target_idx] = 0.0  # Mask the target node's feature to prevent data leakage
            y = torch.tensor([row[self.target_node]], dtype=torch.float32)
            
            # Encode labels based on the task
            if self.task == "classification":
                if self.num_classes is None or self.num_classes == 2:
                    y = y  # Binary classification: labels remain as floats
                else:
                    y = torch.tensor(row[self.target_node], dtype=torch.long)  # Multi-class: labels as integers
            elif self.task == "regression":
                y = y  # Regression: labels remain as floats
            else:
                raise ValueError(f"Unsupported task: {self.task}. Use 'classification' or 'regression'.")
            
            data = Data(x=x, edge_index=self.edge_index, y=y)
            data_list.append(data)
        return data_list
    
    def get_loss_function(self) -> nn.Module:
        """
        Returns the appropriate loss function based on the task.

        Returns:
            nn.Module: The loss function.
        """
        if self.task == "classification":
            if self.num_classes is None or self.num_classes == 2:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task: {self.task}. Use 'classification' or 'regression'.")
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        learning_rate: float = 0.001,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32
    ) -> Dict:
        """
        Trains the GCN model.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            epochs (int, optional): Maximum number of epochs. Defaults to 100.
            patience (int, optional): Early stopping patience. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            Dict: Dictionary containing training and validation accuracies.
        """
        train_dataset = self._prepare_data(train_data)
        val_dataset = self._prepare_data(val_data)
        
        # Determine output dimension based on the task
        if self.task == "classification":
            if self.num_classes is None or self.num_classes == 2:
                output_dim = 1  # Binary classification
            else:
                output_dim = self.num_classes  # Multi-class classification
        elif self.task == "regression":
            output_dim = 1  # Regression
        else:
            raise ValueError(f"Unsupported task: {self.task}. Use 'classification' or 'regression'.")
        
        if self.model is None:
            self.model = GCN(num_features=1, hidden_dim=self.hidden_dim, output_dim=output_dim)
        
        criterion = self.get_loss_function()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
        best_val_loss = float('inf')
        no_improve = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                
                # Reshape output to [batch_size, num_nodes, output_dim]
                batch_size_current = data.num_graphs
                num_nodes = self.adj_mat.shape[0]
                out = out.view(batch_size_current, num_nodes, -1)
                
                # Extract target node's output
                target_out = out[:, self.target_idx, :]  # Shape: [batch_size, output_dim]
                
                if self.task == "classification":
                    if self.num_classes is None or self.num_classes == 2:
                        loss = criterion(target_out.squeeze(), data.y)
                    else:
                        loss = criterion(target_out, data.y)
                elif self.task == "regression":
                    loss = criterion(target_out.squeeze(), data.y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_size_current
            train_loss /= len(train_loader.dataset)
            
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    out = self.model(data.x, data.edge_index)
                    
                    # Reshape output to [batch_size, num_nodes, output_dim]
                    batch_size_current = data.num_graphs
                    num_nodes = self.adj_mat.shape[0]
                    out = out.view(batch_size_current, num_nodes, -1)
                    
                    # Extract target node's output
                    target_out = out[:, self.target_idx, :]  # Shape: [batch_size, output_dim]
                    
                    if self.task == "classification":
                        if self.num_classes is None or self.num_classes == 2:
                            loss = criterion(target_out.squeeze(), data.y)
                            preds = (torch.sigmoid(target_out.squeeze()) >= 0.5).float()
                            correct += (preds == data.y).sum().item()
                            total += data.y.size(0)
                        else:
                            loss = criterion(target_out, data.y)
                            preds = torch.argmax(target_out, dim=1)
                            correct += (preds == data.y).sum().item()
                            total += data.y.size(0)
                    elif self.task == "regression":
                        loss = criterion(target_out.squeeze(), data.y)
                
                    val_loss += loss.item() * batch_size_current
            val_loss /= len(val_loader.dataset)
            
            if self.task == "classification":
                val_acc = correct / total
            else:
                val_acc = None  # Define regression metrics if needed
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if self.task == "classification":
                history['train_acc'].append(None)  # Not calculated here
                history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model.state_dict()
                no_improve = 0
                if self.task == "classification":
                    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
                else:
                    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
        # Load the best model
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        # Calculate final accuracies
        train_preds = self.predict(train_data)
        val_preds = self.predict(val_data)
        
        if self.task == "classification":
            train_acc = accuracy_score(train_data[self.target_node], train_preds)
            val_acc = accuracy_score(val_data[self.target_node], val_preds)
        else:
            train_acc = None
            val_acc = None
        
        return {"train_acc": train_acc, "val_acc": val_acc, "history": history}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the input data.

        Args:
            X (pd.DataFrame): Input dataset.

        Returns:
            np.ndarray: Predicted labels or values.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        dataset = self._prepare_data(X)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for data in loader:
                out = self.model(data.x, data.edge_index)
                
                # Reshape output to [batch_size, num_nodes, output_dim]
                batch_size_current = data.num_graphs
                num_nodes = self.adj_mat.shape[0]
                out = out.view(batch_size_current, num_nodes, -1)
                
                # Extract target node's output
                target_out = out[:, self.target_idx, :]  # Shape: [batch_size, output_dim]
                
                if self.task == "classification":
                    if self.num_classes is None or self.num_classes == 2:
                        preds_tensor = torch.sigmoid(target_out.squeeze())
                        pred_labels = (preds_tensor >= 0.5).float()
                        preds.extend(pred_labels.cpu().numpy())
                    else:
                        preds_tensor = out
                        pred_labels = torch.argmax(target_out, dim=1)
                        preds.extend(pred_labels.cpu().numpy())
                elif self.task == "regression":
                    pred_values = target_out.squeeze().cpu().numpy()
                    preds.extend(pred_values)
        return np.array(preds)
    
# Example usage
if __name__ == "__main__":
    from src.data_processing.data_loader import load_synthetic_data
    from src.data_processing.preprocessor import preprocess_data
    from src.utils.graph_utils import get_adjacency_matrix
    
    dataset = "alarm"
    df = load_synthetic_data(dataset=dataset)
    train, val, test = preprocess_data(df)
    
    asia_edges = get_adjacency_matrix(dataset)
    target_node = "BP"
    num_classes = df[target_node].nunique()
    
    gcn_model = GCNBaseline(target_node=target_node, adj_mat=asia_edges, task="classification", num_classes=num_classes)
    gcn_metrics = gcn_model.train(train, val)
    gcn_test_preds = gcn_model.predict(test)
    gcn_test_acc = accuracy_score(test[target_node], gcn_test_preds)
    
    print(f"Test Metrics for GCN predicting {target_node} in {dataset}:")
    print(f"Train Accuracy: {gcn_metrics['train_acc']:.3f}")
    print(f"Val Accuracy: {gcn_metrics['val_acc']:.3f}")
    print(f"Test Accuracy: {gcn_test_acc:.3f}")
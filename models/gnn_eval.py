import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from models.utils import split_data, calculate_metrics
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Define the GNN Model
class SimpleGNN(nn.Module):
    def __init__(self, input_size, output_size, task):
        """
        A simple Graph Convolutional Network (GCN).
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (1 for regression, num_classes for classification).
            task (str): Task type ('classification' or 'regression').
        """
        super(SimpleGNN, self).__init__()
        self.task = task
        self.conv1 = GCNConv(input_size, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, output_size)
        self.activation = nn.Softmax(dim=1) if task == "classification" else nn.Identity()

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return self.activation(x)

def train_and_evaluate_gnn(data, target_col, task, variation, file_name):
    """
    Train and evaluate a simple GNN model on a given graph dataset.
    
    Args:
        graph_data (Data): PyTorch Geometric data object containing the graph.
        target_col (str): The name of the target column.
        task (str): The type of task ('classification' or 'regression').
        variation (str): The dataset variation (e.g., 'base', 'flipped').
        file_name (str): The name of the dataset file.

    Returns:
        dict: A dictionary containing evaluation metrics and metadata.
    """

    # Separate node features and target.
    node_features = data.drop(columns=[target_col]).values
    target = data[target_col].values
    
    # Convert them to tensors, adjusting for classification if needed.
    node_features = torch.tensor(node_features, dtype=torch.float32)
    if task == 'classification':
        target = torch.tensor(target, dtype=torch.long)
    else:
        target = torch.tensor(target, dtype=torch.float32)
    
    # Train, val, test split with sklearn.
    train_idx, test_idx = train_test_split(
        range(data.shape[0]), test_size=0.4, random_state=0
    )
    val_idx, test_idx = train_test_split(
        test_idx, test_size=0.5, random_state=0
    )
    
    # Build one fully connected edge index for the entire dataset.
    def build_fully_connected_edge_index(num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append([i, j])
                edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    edge_index = build_fully_connected_edge_index(data.shape[1])
    
    # Create separate Data objects for each split, using the same edge_index.
    train_data = Data(x=node_features[train_idx], y=target[train_idx], edge_index=edge_index)
    val_data = Data(x=node_features[val_idx], y=target[val_idx], edge_index=edge_index)
    test_data = Data(x=node_features[test_idx], y=target[test_idx], edge_index=edge_index)
    
    # Define model, loss, and optimizer (similar to causal_gnn.py).
    input_size = node_features.shape[1]
    output_size = len(torch.unique(target)) if task == "classification" else 1
    model = SimpleGNN(input_size, output_size, task)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping setup.
    min_delta = 0.0005
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    epoch = 0
    
    # Training loop with early stopping.
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index)
            val_loss = criterion(val_out, val_data.y).item()
        
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        # Stop if no improvement for 'patience' epochs
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load the best model state, evaluate on test set.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.edge_index)
        if task == 'classification':
            predictions = torch.argmax(test_out, dim=1).numpy()
        else:
            predictions = test_out.numpy()
    y_test = test_data.y.numpy()
    
    # Calculate metrics, add metadata, return.
    metrics = calculate_metrics(y_test, predictions, task)
    metrics.update({"Model": "GNN", "Variation": variation, "File": file_name})
    return metrics


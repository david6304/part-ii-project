import torch
import torch.nn as nn
import torch.optim as optim
from models.utils import split_data, calculate_metrics
import pandas as pd

# Define the DNN Model
class SimpleDNN(nn.Module):
    def __init__(self, input_size, output_size, task):
        """
        A simple feedforward neural network.
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (1 for regression, num_classes for classification).
            task (str): Task type ('classification' or 'regression').
        """
        super(SimpleDNN, self).__init__()
        self.task = task
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.activation = nn.Softmax(dim=1) if task == "classification" else nn.Identity()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.activation(x)

def train_and_evaluate_dnn(data, target_col, task, variation, file_name):
    """
    Train and evaluate a simple DNN model on a given dataset, using a separate validation set.
    
    Args:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target_col (str): The name of the target column.
        task (str): The type of task ('classification' or 'regression').
        variation (str): The dataset variation (e.g., 'base', 'flipped').
        file_name (str): The name of the dataset file.

    Returns:
        dict: A dictionary containing evaluation metrics and metadata.
    """

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = split_data(data, target_col, test_size=0.4)  # 60% train, 40% temp
    X_val, X_test, y_val, y_test = split_data(
        pd.concat([X_temp, y_temp], axis=1), target_col, test_size=0.5
    )  # 20% validation, 20% test

    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Adjust target shape for classification
    if task == "classification":
        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()

    # Define model, loss, and optimizer
    input_size = X_train.shape[1]
    output_size = len(y_train.unique()) if task == "classification" else 1
    model = SimpleDNN(input_size, output_size, task)

    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 5
    min_delta = 0.0005
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # To store the best model's state_dict

    # Training loop with early stopping
    for epoch in range(1000):  # Large max epoch limit
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        # Check for improvement
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            patience_counter += 1

        # Stop if no improvement for 'patience' epochs
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load the best model before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        if task == "classification":
            predictions = torch.argmax(predictions, dim=1)

    # Convert predictions and targets to numpy
    predictions = predictions.numpy()
    y_test = y_test.numpy()

    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions, task)
    metrics.update({"Model": "DNN", "Variation": variation, "File": file_name})
    
    del model
    del optimizer

    return metrics
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import bnlearn as bn
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load example dataset
X = bn.import_example('asia', n=10000)

# Learn causal graph from the entire dataset
G = grasp(X)
print(G)

# Convert GRaSP output to edge index format for PyTorch Geometric
edge_list = [(i, j) for i in range(len(G.graph)) for j in range(len(G.graph[i])) if G.graph[i][j] == -1]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Prepare node features and target
node_features = torch.tensor(X.drop(columns=['dysp']).values, dtype=torch.float)  # Exclude target from input features
target = torch.tensor(X['dysp'].values, dtype=torch.long)  # Target variable for classification

# Train-test split
train_idx, test_idx = train_test_split(range(X.shape[0]), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)  # Further split test set into validation and test sets

# Create PyTorch Geometric data objects for training, validation, and testing
train_data = Data(x=node_features[train_idx], edge_index=edge_index)
train_data.y = target[train_idx]

val_data = Data(x=node_features[val_idx], edge_index=edge_index)
val_data.y = target[val_idx]

test_data = Data(x=node_features[test_idx], edge_index=edge_index)
test_data.y = target[test_idx]

# Define a simple GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, define loss and optimizer
input_dim = node_features.shape[1]
output_dim = len(torch.unique(target))  # Number of unique classes in the target
model = GNN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0
epoch = 0
best_model_state = None  # To store the best model state

while patience_counter < patience:
    model.train()
    optimizer.zero_grad()
    out = model(train_data)  # Raw logits
    loss = criterion(out, train_data.y)  # Use CrossEntropyLoss
    loss.backward()
    optimizer.step()
    
    # Validation loss
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_loss = criterion(val_out, val_data.y)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    
    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()  # Save the best model state
    else:
        patience_counter += 1
    
    epoch += 1

# Load the best model for testing
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_out = model(test_data)  # Raw logits
    test_pred = torch.argmax(test_out, dim=1)  # Convert logits to class predictions

    # Calculate accuracy
    accuracy = accuracy_score(test_data.y.numpy(), test_pred.numpy())

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize the learned causal graph
pyd = GraphUtils.to_pydot(G)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

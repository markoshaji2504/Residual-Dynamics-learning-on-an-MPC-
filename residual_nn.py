import torch
import torch.nn as nn
import numpy as np 
import math 

#load the data
data = np.load('data/residuals.npz')
states = data['states']
inputs = data['inputs']
#discrepnacy levels
discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]


class ResidualNN(nn.Module):             # define a new neural network, built on PyTorch's base class
    def __init__(self):                  # constructor: defines what layers the network has
        super().__init__()               # required setup call for any class built on nn.Module
        self.net = nn.Sequential(        # chain layers together in order, applied one after another
            nn.Linear(3, 32),            # input layer: 3 numbers in (theta, theta_dot, u) -> 32 features out
            nn.ReLU(),                   # nonlinearity: without this, stacked linear layers would collapse into one
            nn.Linear(32, 32),           # hidden layer: 32 features in -> 32 features out
            nn.ReLU(),                   # another nonlinearity after the hidden layer
            nn.Linear(32, 2)             # output layer: 32 features in -> 2 numbers out (predicted residual)
        )

    def forward(self, x):                # defines what happens when the network is actually called with input x
        return self.net(x)               # pass x through the full chain of layers, return the result


results = {}          # will store validation loss per discrepancy level
trained_models = {}   # will store the trained NN for each level

for level in discrepancy_levels:
    # --- build data for this level ---
    residual_target = data[f'residual_{level}']
    X = np.hstack([states, inputs.reshape(-1, 1)])
    y = residual_target

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # --- split ---
    n_total = X_tensor.shape[0]
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    # --- fresh model for this level ---
    model = ResidualNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- train ---
    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    # --- evaluate ---
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_fn(val_predictions, y_val)

    print(f"Discrepancy {level}: Final Validation Loss = {val_loss.item():.6f}")

    results[level] = val_loss.item()
    trained_models[level] = model

    for level, mse in results.items():
       rmse = math.sqrt(mse)
       print(f"Discrepancy {level}: RMSE = {rmse:.4f}")
import torch
import torch.nn as nn
import numpy as np 

#load the data
data = np.load('data/residuals.npz')
states = data['states']
inputs = data['inputs']


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


level = 0.30                                    # choose which discrepancy level's data to train on first
residual_target = data[f'residual_{level}']     # look up the saved residual array for that specific level

X = np.hstack([states, inputs.reshape(-1, 1)])   # combine states + inputs into one (1800, 3) input array
y = residual_target                              # the target the NN should learn to predict: (1800, 2)

X_tensor = torch.tensor(X, dtype=torch.float32)   # convert inputs into PyTorch's tensor format
y_tensor = torch.tensor(y, dtype=torch.float32)   # convert targets into PyTorch's tensor format

#new code for the actual training and validation sets 
n_total = X_tensor.shape[0]
n_train = int(0.8 * n_total)

perm = torch.randperm(n_total)
train_idx, val_idx = perm[:n_train], perm[n_train:]

X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]



model = ResidualNN()                                    # create an instance of the network (untrained, random weights)
loss_fn = nn.MSELoss()                                   # mean squared error: measures how far predictions are from targets
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer: adjusts the network's weights during training

n_epochs = 500                          # how many times to pass through the entire dataset during training

for epoch in range(n_epochs):
    optimizer.zero_grad()               # clear gradients from the previous step (PyTorch accumulates them by default)
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    loss.backward()                     # compute gradients: how much each weight contributed to the error
    optimizer.step()                    # adjust the weights slightly, to reduce the error

    if epoch % 50 == 0:                 # every 50 epochs, print progress so you can watch training happen
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    val_predictions = model(X_val)
    val_loss = loss_fn(val_predictions, y_val)
    print(f"Final Validation Loss: {val_loss.item():.6f}")
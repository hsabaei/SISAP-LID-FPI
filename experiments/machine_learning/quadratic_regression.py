import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import time

from estimators import LIDEstimators

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 10 + 3*x - (x**2) + 0.2 * np.random.randn(100, 1)

# Shuffle the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Split indices for training and validation
train_idx = idx[:80]
val_idx = idx[80:]

# Generate train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Convert data to PyTorch Tensors
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Define the Quadratic Regression model
class QuadraticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a * (x**2) + self.b * x + self.c

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)  # Note: Swapped yhat and y for correct order
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

# Initialize the model, loss function, and optimizer
model = QuadraticRegression().to(device)

lr = 1e-1
n_epochs = 2000

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)
lid_estimator = LIDEstimators(device=device)

# Create the train_step function
train_step = make_train_step(model, loss_fn, optimizer)
losses = []
lid_FIE_data = []
lid_Bay_data = []
lid_IR_data = []
lid_LL_data = []
times_FIE = 0
times_Bay = 0
times_IR = 0
times_LL = 0

# Define the size k for the parameter history
k = 23

# Initialize arrays to store the parameter history
param_a_array = np.zeros(k)
param_b_array = np.zeros(k)
param_c_array = np.zeros(k)

Num0 = 0
Num1 = 0
Den0 = 0
Den1 = 0
# Training loop
for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

    # Get current parameter values
    current_a = model.a.item()
    current_b = model.b.item()
    current_c = model.c.item()

    # Shift the arrays to the left and discard the oldest value
    param_a_array[:-1] = param_a_array[1:]
    param_b_array[:-1] = param_b_array[1:]
    param_c_array[:-1] = param_c_array[1:]

    # Store the current values at the end of the arrays
    param_a_array[-1] = current_a
    param_b_array[-1] = current_b
    param_c_array[-1] = current_c

    if epoch >= k - 1:

        #FIE LID estimation for parameter 'a'
        start_time = time.time()
        LID_FIE_a = lid_estimator.compute_FIE_LID(param_a_array)
        lid_FIE_data.append([epoch + 1, LID_FIE_a])
        times_FIE += time.time() - start_time

        # Bayesian LID estimation for parameter 'a'
        start_time = time.time()
        LID_Bay_a, Num1, Den1 = lid_estimator.compute_Bayesian_LID(param_a_array, Num0, Den0)
        # Update cumulative numerator and denominator for next iteration
        Num0 = Num1
        Den0 = Den1
        lid_Bay_data.append([epoch + 1, LID_Bay_a])
        times_Bay += time.time() - start_time

        #IR LID estimation for parameter 'a'
        start_time = time.time()
        LID_IR_a = lid_estimator.compute_IR_LID(param_a_array)
        lid_IR_data.append([epoch + 1, LID_IR_a])
        times_IR += time.time() - start_time

        #Log-Log LID estimation for parameter 'a'
        start_time = time.time()
        LID_LL_a = lid_estimator.compute_LL_LID(param_a_array[0:k-1], param_a_array[1:k], lr=0.01, n_epochs=2000)
        lid_LL_data.append([epoch + 1, LID_LL_a])
        times_LL += time.time() - start_time

# Check the learned parameters a, b, and c
print("\nFinal learned parameters:")
print(f"a: {model.a.item():.4f}")
print(f"b: {model.b.item():.4f}")
print(f"c: {model.c.item():.4f}")

true_lid = 1.0

# FIE:
lid_FIE_values = np.array([data[1] for data in lid_FIE_data[5:]])
mean_lid_FIE = np.mean(lid_FIE_values)
var_lid_FIE = np.var(lid_FIE_values)
square_bias_FIE = (mean_lid_FIE - true_lid)**2
msne_FIE =(square_bias_FIE + var_lid_FIE) / (true_lid**2)

# Bayesian:
lid_Bay_values = np.array([data[1] for data in lid_Bay_data[5:]])
mean_lid_Bay = np.mean(lid_Bay_values)
var_lid_Bay = np.var(lid_Bay_values)
square_bias_Bay = (mean_lid_Bay - true_lid)**2
msne_Bay = (square_bias_Bay + var_lid_Bay) / (true_lid**2)

# IR:
lid_IR_values = np.array([data[1] for data in lid_IR_data[5:]])
valid_lid_IR_values = lid_IR_values[np.isfinite(lid_IR_values)]
if valid_lid_IR_values.size > 0:
    mean_lid_IR = np.mean(valid_lid_IR_values)
    var_lid_IR = np.var(valid_lid_IR_values)
    square_bias_IR = (mean_lid_IR - true_lid)**2
    msne_IR = (square_bias_IR + var_lid_IR) / (true_lid**2)
else:
    # If no valid IR values, set them to NaN
    mean_lid_IR = np.nan
    var_lid_IR = np.nan
    square_bias_IR = np.nan
    msne_IR = np.nan

# Log-Log:
lid_LL_values = np.array([data[1] for data in lid_LL_data[5:]])
mean_lid_LL = np.mean(lid_LL_values)
var_lid_LL = np.var(lid_LL_values)
square_bias_LL = (mean_lid_LL - true_lid)**2
msne_LL = (square_bias_LL + var_lid_LL) / (true_lid**2)

results_dict = {
    "Estimator": ["FIE", "Bayesian", "IR", "Log-Log"],
    "Mean": [mean_lid_FIE, mean_lid_Bay, mean_lid_IR, mean_lid_LL],
    "Variance": [var_lid_FIE, var_lid_Bay, var_lid_IR, var_lid_LL],
    "Square Bias": [square_bias_FIE, square_bias_Bay, square_bias_IR, square_bias_LL],
    "MSNE": [msne_FIE, msne_Bay, msne_IR, msne_LL],
    "Time(s)": [times_FIE, times_Bay, times_IR, times_LL]
}

df = pd.DataFrame(results_dict)
print(df.to_string(index=False))  

# Create DataFrames for lid_FIE_data and lid_Bay_data
lid_FIE_df = pd.DataFrame(lid_FIE_data, columns=['Epoch', 'LID_FIE'])
lid_Bay_df = pd.DataFrame(lid_Bay_data, columns=['Epoch', 'LID_Bayesian'])
lid_IR_df = pd.DataFrame(lid_IR_data, columns=['Epoch', 'LID_IR'])
lid_LL_df = pd.DataFrame(lid_LL_data, columns=['Epoch', 'LID_Log-Log'])

plt.figure(figsize=(8, 6))

plt.plot(lid_IR_df['Epoch'], lid_IR_df['LID_IR'],
         marker='*', linestyle='-.', color='green',
         label='LID_IR_a', linewidth=0.3, markersize=3, markeredgewidth=0.3)
plt.plot(lid_LL_df['Epoch'], lid_LL_df['LID_Log-Log'],
         marker='o', linestyle=':', color='pink',
         label='LID_LL_a', linewidth=0.3, markersize=3, markeredgewidth=0.3)

plt.plot(lid_FIE_df['Epoch'], lid_FIE_df['LID_FIE'],
         marker='o', linestyle='-', color='blue',
         label='LID_FIE_a', linewidth=0.5, markersize=3, markeredgewidth=0.3)
plt.plot(lid_Bay_df['Epoch'], lid_Bay_df['LID_Bayesian'],
         marker='x', linestyle='--', color='red',
         label='LID_Bay_a', linewidth=0.5, markersize=2, markeredgewidth=0.3)
plt.axhline(y=1, color='black', linestyle='-', linewidth=1, label='True LID')

plt.title('LID Estimations for Parameter "a1" Over Epochs (k=10)')
plt.xlabel('Epoch')
plt.ylabel('LID Value')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Plotting
plt.plot(lid_IR_df['Epoch'], lid_IR_df['LID_IR'],
         marker='*', linestyle='-.', color='green',
         label='LID_IR_a', linewidth=0.3, markersize=3, markeredgewidth=0.3)
plt.plot(lid_LL_df['Epoch'], lid_LL_df['LID_Log-Log'],
         marker='o', linestyle=':', color='pink',
         label='LID_LL_a', linewidth=0.3, markersize=3, markeredgewidth=0.3)

plt.plot(lid_FIE_df['Epoch'], lid_FIE_df['LID_FIE'],
         marker='o', linestyle='-', color='blue',
         label='LID_FIE_a', linewidth=0.5, markersize=3, markeredgewidth=0.3)
plt.plot(lid_Bay_df['Epoch'], lid_Bay_df['LID_Bayesian'],
         marker='x', linestyle='--', color='red',
         label='LID_Bay_a', linewidth=0.5, markersize=2, markeredgewidth=0.3)
plt.axhline(y=1, color='black', linestyle='-', linewidth=1, label='True LID')

plt.title('LID Estimations for Parameter "a1" Over Epochs (k=10)')
plt.xlabel('Epoch')
plt.ylabel('LID Value')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.ylim(0, 3)
#plt.xlim(0, 200)
plt.show()

import numpy as np
import torch
import time
from scipy.stats import norm
from estimators import LIDEstimators

device = 'cuda' if torch.cuda.is_available() else 'cpu'
U = lambda x: x - ((x**2 - 4*x + 4)/(2* x - 4))

lid_estimator = LIDEstimators(device=device)

# Lists to collect results
fpi_values = []
ircoe_values = []
ll_bs_values = []
ll_ba_values = []
times_fpi = []
times_ircoe = []
times_ll_bs = []
times_ll_ba = []

t= 30 #Number of iteration
k = 13 #Number of samples = 10

for _ in range(1):
    x = np.zeros(t)
    start_time = time.time()

# Set initial value
    x[0] = float(input("Enter the starting point: "))
    for i in range(0, t-1):
        x[i+1] = U(x[i])
    print(x)

    X = x[t-1-k:t-1]
    G = x[t-k:t]

    start_time = time.time()
    fpi_result = lid_estimator.compute_FIE_LID(X)
    fpi_values.append(fpi_result)
    times_fpi.append(time.time() - start_time)

    start_time = time.time()
    ircoe_result = lid_estimator.compute_IR_LID(X)
    ircoe_values.append(ircoe_result)
    times_ircoe.append(time.time() - start_time)

    start_time = time.time()
    ll_bs_result = lid_estimator.compute_LL_LID(X,G, lr=0.1, n_epochs=100)
    ll_bs_values.append(ll_bs_result)
    times_ll_bs.append(time.time() - start_time)

    start_time = time.time()
    ll_ba_result = lid_estimator.compute_LL_LID(X, G,lr=0.01, n_epochs=2000)
    ll_ba_values.append(ll_ba_result)
    times_ll_ba.append(time.time() - start_time)

"""
print(f"The FPI estimator value is: {fpi_values}")
print(f"The IRCOE estimator value is: {ircoe_values}")
print(f"The log-log (BS) estimator value is: {np.abs(ll_bs_values)}")
print(f"The log-log (BA) estimator value is: {np.abs(ll_ba_values)}")
"""
# Calculate statistics
def calculate_stats(values):
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

fpi_mean, fpi_std = calculate_stats(fpi_values)
ircoe_mean, ircoe_std = calculate_stats(ircoe_values)
ll_bs_mean, ll_bs_std = calculate_stats(ll_bs_values)
ll_ba_mean, ll_ba_std = calculate_stats(ll_ba_values)

times_fpi_mean = np.mean(times_fpi)
times_ircoe_mean = np.mean(times_ircoe)
times_ll_bs_mean = np.mean(times_ll_bs)
times_ll_ba_mean = np.mean(times_ll_ba)

# Print the results
print(f"FPI: Mean = {fpi_mean}, Std = {fpi_std}")
print(f"IRCOE: Mean = {ircoe_mean}, Std = {ircoe_std}")
print(f"Log-Log (BS): Mean = {ll_bs_mean}, Std = {ll_bs_std}")
print(f"Log-Log (BA): Mean = {ll_ba_mean}, Std = {ll_ba_std}")

print(f"Time FPI: Mean = {times_fpi_mean}")
print(f"Time IRCOE: Mean = {times_ircoe_mean}")
print(f"Time Log-Log (BS): Mean = {times_ll_bs_mean}")
print(f"Time Log-Log (BA): Mean = {times_ll_ba_mean}")

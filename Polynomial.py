import numpy as np
import torch
import time
from scipy.stats import norm
from estimators import LIDEstimators


# Define parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
f = lambda x: 4 * x**5 +  x**3 + 1  # Define the function
a, b = 0 , 2.5  # Define the interval

lid_estimator = LIDEstimators(device='cuda')

# Lists to collect results
fpi_values = []
ircoe_values = []
ll_bs_values = []
ll_ba_values = []
times_fpi = []
times_ircoe = []
times_ll_bs = []
times_ll_ba = []

k_values = [32, 64, 128, 256]

for k in k_values:
    # Uniform Intervals
    Uni_Int_points = np.linspace(a, b, k)
    y_UniInt_noiseless = f(Uni_Int_points)
    X = Uni_Int_points[::-1]
    Y = y_UniInt_noiseless[::-1]

    start_time = time.time()
    fpi_result = lid_estimator.compute_GIE_LID(X, Y)
    fpi_values.append(fpi_result)
    times_fpi.append(time.time() - start_time)

    start_time = time.time()
    ircoe_result = lid_estimator.compute_IR_LID(Y)
    ircoe_values.append(ircoe_result)
    times_ircoe.append(time.time() - start_time)

    start_time = time.time()
    ll_bs_result = lid_estimator.compute_LL_LID(X,Y, lr=0.1, n_epochs=100)
    ll_bs_values.append(ll_bs_result)
    times_ll_bs.append(time.time() - start_time)

    start_time = time.time()
    ll_ba_result = lid_estimator.compute_LL_LID(X,Y, lr=0.01, n_epochs=2000)
    ll_ba_values.append(ll_ba_result)
    times_ll_ba.append(time.time() - start_time)


print(f"The FPI estimator values are: {fpi_values}")
print(f"The FPI estimator times are: {times_fpi}")
print(f"The IRCOE estimator values are: {ircoe_values}")
print(f"The IRCOE estimator times are: {times_ircoe}")
print(f"The log-log (BS) estimator values are: {ll_bs_values}")
print(f"The log-log (BS) estimator times are: {times_ll_ba}")
print(f"The log-log (BA) estimator values are: {ll_ba_values}")
print(f"The log-log (BA) estimator times are: {times_ll_bs}")

# Calculate statistics
def calculate_stats(values):
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

times_fpi_mean = np.mean(times_fpi)
times_ircoe_mean = np.mean(times_ircoe)
times_ll_bs_mean = np.mean(times_ll_bs)
times_ll_ba_mean = np.mean(times_ll_ba)

print(f"Time FPI: Mean = {times_fpi_mean}")
print(f"Time IRCOE: Mean = {times_ircoe_mean}")
print(f"Time Log-Log (BS): Mean = {times_ll_bs_mean}")
print(f"Time Log-Log (BA): Mean = {times_ll_ba_mean}")


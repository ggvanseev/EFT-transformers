import torch
import numpy as np
from self_attention_nn import NN

import os

# To avoid matplotlib errors
os.makedirs("tmp", exist_ok=True)
os.environ["MPLCONFIGDIR"] = "/tmp"

from datetime import datetime
import yaml
from theoretical_forward_equation import G_MLP, G_MHSA
from plotting import plot_correlation_function_comparison, plot_histogram_comparison

# This file performes a comparison between the theoretical forward equation
# for the covariance matrix of the l-th layer to a neural network layer.
# Index of corellation function
# an integor, or 'avg', which means that we sum over all indices
# and divide by the total number of indices
delta = 0
t = 0
i = 0

# Provide existing results directory, if None, create a new one with
# hyperparameters to set below
dir = None

# Create results
if dir is None:
    ############################################
    ############### Settings ###################
    ############################################
    # Choose hyperparameters
    N_net = int(2e4)  # Number of neural networks
    d = 1  # Number of samples in the batch
    n_t = 1  # Number of tokens
    n_in = 1  # Number of input features
    n = 10  # Number of features/neurons in hidden/output layers
    n_h = 1  # Number of attention heads
    num_layers = 3  # Total number of layers in the stack
    # Width of the Gaussian distribution for initialization
    # For the naming convetion see the added paper
    W_std = 0.5
    E_std = 0.5
    Q_std = 0.5

    # Theoretical choices:
    # n_invariance flag Determines whether the weights
    # are made invariant with respect to the number of
    # , True is yes, False is no.
    invariance_flags = {"n": True, "n_t": False, "n_h": True}

    # forward_use_std_flag, if True, the forward equation uses
    # the std instead of the variance. Note that this is
    # erronous, but succesful when comparing to numerical results
    # If False, the forward equation uses the variance (as it should)
    forward_use_std_flag = False

    # Type of the neural network
    NN_type = "MHSA"  # "MHSA", or "MLP"

    # Storage settings
    # If None, will use a time stamp
    dir_name = None
    # Store intermediate results, just for debug purposes
    store_intermediate_flag = True

    #############################################
    ############## End of Settings ##############
    #############################################

    # A random input tensor, shape (N_net, d, n_t, n_in)
    x = torch.stack([torch.randn(d, n_t, n_in)] * N_net)

    # Perform NN training for several models
    NN_result = np.zeros((num_layers, N_net, d, n_t, n))

    stack = NN(
        n,
        n_h,
        n_t,
        n_in,
        num_layers,
        W_std=W_std,
        E_std=E_std,
        Q_std=Q_std,
        invariance_flags=invariance_flags,
        N_net=N_net,
        NN_type=NN_type,
    )

    _ = stack(x, store_intermediate_flag=store_intermediate_flag)

    if store_intermediate_flag:
        NN_result = np.array(stack.layer_outputs)

    del stack

    # Save the results

    # Save the results in a directory with a time marking
    if dir_name is None:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d-%H%M-%S")
        dir = f"/data/theorie/gseevent/edinburgh/results/{formatted_time}"
    else:
        dir = f"/data/theorie/gseevent/edinburgh/results/{dir_name}"

    os.makedirs(dir, exist_ok=True)

    hyperparameters = {
        "N_net": N_net,
        "d": d,
        "n_t": n_t,
        "n_in": n_in,
        "n": n,
        "n_h": n_h,
        "num_layers": num_layers,
        "W_std": W_std,
        "E_std": E_std,
        "Q_std": Q_std,
        "n_invariance_flag": invariance_flags["n"],
        "nt_invariance_flag": invariance_flags["n_t"],
        "nh_invariance_flag": invariance_flags["n_h"],
        "forward_use_std_flag": forward_use_std_flag,
        "NN_type": NN_type,
        "delta": delta,
        "t": t,
        "i": i,
    }

    with open(f"{dir}/hyperparameters.yaml", "w") as file:
        yaml.dump(hyperparameters, file)

    np.save(f"{dir}/NN_result.npy", NN_result)
    np.save(f"{dir}/random_input_data_x.npy", x.numpy())
else:
    # Load the results
    NN_result = np.load(f"{dir}/NN_result.npy")
    x = np.load(f"{dir}/random_input_data_x.npy")

    with open(f"{dir}/hyperparameters.yaml", "r") as file:
        hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

    # Get necessary hyperparameters
    N_net = hyperparameters["N_net"]
    d = hyperparameters["d"]
    n_t = hyperparameters["n_t"]
    n_in = hyperparameters["n_in"]
    n = hyperparameters["n"]
    n_h = hyperparameters["n_h"]
    num_layers = hyperparameters["num_layers"]
    NN_type = hyperparameters["NN_type"]
    E_std = hyperparameters["E_std"]
    Q_std = hyperparameters["Q_std"]
    W_std = hyperparameters["W_std"]
    invariance_flags = dict()
    invariance_flags["n"] = hyperparameters["n_invariance_flag"]
    invariance_flags["n_t"] = hyperparameters["nt_invariance_flag"]
    invariance_flags["n_h"] = hyperparameters["nh_invariance_flag"]
    n_invariance_flag = hyperparameters["n_invariance_flag"]
    forward_use_std_flag = hyperparameters["forward_use_std_flag"]

    # Update for plotting the current run
    hyperparameters["delta"] = delta
    hyperparameters["t"] = t
    hyperparameters["i"] = i


# Calculate the corellation functions
def correlation_function_NN(
    r_1: np.ndarray,
    power_1: int,
    r_2: np.ndarray = None,
    power_2: int = 0,
    expected_shape=(num_layers, N_net, d, n_t, n),
):
    r"""
    Calculates the correlation function over an ensemble of neural networks.
    Assumes a specific dataset, token and neural index, i.e.
    r_1[N_i] = r_{\delta_1 t_1 i}^ell
    Note that for n or m=2 or odd, this equals the connected 'diagrams'.
    input:
    r_1: np.ndarray, shape=(layers,N_net, d,n_t,n)
    r_2: np.ndarray, shape=(layers,N_net, d,n_t,n)
    power_1: int, the power to which to raise r_1
    power_2: int, the power to which to raise r_2
    return:
    r: np.ndarray, shape=(layers, d, n_t, n)
    """

    # Ensure correct input
    assert r_1.shape == expected_shape

    if r_2:
        return np.mean(r_1**power_1 * r_2**power_2, axis=1)
    else:
        return np.mean(r_1**power_1, axis=1)


# Get the index or average over all indices
def get_index_or_avg(plot_type: str, NN_result: np.ndarray, delta, t, i, d: int = d):
    r"""
    Returns the correlation function for a specific index or the average over all indices.
    input:
    plot_type: str, 'correlation' or 'histogram'
    NN_result:
    np.ndarray, shape=(layers,d,n_t,n) if plot_type='correlation'
    np.ndarray, shape=(layers, N_net,d,n_t,n) if plot_type='histogram'
    delta: int, the delta index, or 'avg'
    t: int, the t index, or 'avg'
    i: int, the i index, or 'avg'
    output:
    NN_result: np.ndarray, shape=(layers)
    """
    if plot_type == "correlation":
        axes = {3: i, 2: t, 1: delta}
    elif plot_type == "histogram":
        axes = {4: i, 3: t, 2: delta}
    else:
        raise ValueError("plot_type must be 'correlation' or 'histogram'")

    for axis, value in axes.items():
        if value == "avg":
            NN_result = np.mean(NN_result, axis=axis)
        else:
            NN_result = np.take(NN_result, indices=value, axis=axis)

    return NN_result


NN_result_r1 = get_index_or_avg(
    "correlation", correlation_function_NN(NN_result, 1), delta, t, i
)
NN_result_r2 = get_index_or_avg(
    "correlation", correlation_function_NN(NN_result, 2), delta, t, i
)
NN_result_r3 = get_index_or_avg(
    "correlation", correlation_function_NN(NN_result, 3), delta, t, i
)

NN_result_hist = get_index_or_avg("histogram", NN_result, delta, t, i)

if NN_type == "MHSA":
    # G shape=(n_layers, d, d, t, t)
    G = G_MHSA(
        num_layers,
        Q_std,
        E_std,
        n,
        n_h,
        W_std,
        x[0],
        invariance_flags=invariance_flags,
        forward_use_std_flag=forward_use_std_flag,
    )

    if t == "avg":
        G = np.mean(G, axis=4)
        G = np.mean(G, axis=3)
    else:
        G = G[:, :, :, t, t]

    if delta == "avg":
        G = np.mean(G, axis=2)
        G = np.mean(G, axis=1)
    else:
        G = G[:, delta, delta]
elif NN_type == "MLP":
    G = G_MLP(
        num_layers,
        n,
        W_std,
        x[0, :, 0, :],
        n_independent_flag=invariance_flags["n"],
        forward_use_std_flag=forward_use_std_flag,
    )

    if delta == "avg":
        G = np.mean(G, axis=2)
        G = np.mean(G, axis=1)
    else:
        G = G[:, delta, delta]
else:
    raise ValueError("NN_type must be 'MHSA' or 'MLP'")


# Compare the results via plotting
plot_histogram_comparison(
    NN_result_hist,
    x=x[0],
    var_theory=G,
    figname=f"{NN_type}_histogram_d{delta}-t{t}-i{i}.png",
    dir=dir,
    hyperparameters=hyperparameters,
)

plot_correlation_function_comparison(
    NN_result_r1,
    NN_result_r2,
    NN_result_r3,
    corellation_G=G,
    figname=f"{NN_type}_correlation_d{delta}-t{t}-i{i}.png",
    dir=dir,
    hyperparameters=hyperparameters,
    delta=delta,
    t=t,
    i=i,
)

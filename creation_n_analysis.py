import torch
import numpy as np
from self_attention_nn import NN

import os

# To avoid matplotlib errors
os.makedirs("tmp", exist_ok=True)
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from self_attention_forward_equation import G

# This file performes a comparison between the theoretical forward equation
# for the covariance matrix of the l-th layer to a neural network layer.
# Index of corellation function
# Avg means that we sum over all indices and divide by the number of indices
delta = "avg"
t = "avg"
i = 0

# Provide existing results directory, if None, create a new one with
# hyperparameters to set below
dir = "/data/theorie/gseevent/edinburgh/results/1126-1923-52"  # "/data/theorie/gseevent/edinburgh/results/varying_t_index"


# Create results
if dir is None:
    # Choose hyperparameters
    N_net = 10  # Number of neural networks
    d = 4  # Number of samples in the batch
    n_t = 20  # Number of tokens
    n_in = 1  # Number of input features
    n = 20  # Number of features/neurons in hidden/output layers
    n_h = 1  # Number of attention heads
    num_layers = 5  # Total number of layers in the stack
    # Width of the Gaussian distribution for initialization
    weight_input_std = 0.5
    weight_E_std = 0.5
    weight_Q_std = 0.5
    n_invariance_flag = True
    # n_invariance flag Determines whether the input weights
    # are made invariant, True is yes, False is no.

    # Type of the neural network
    NN_type = "multihead-self-attention"  # "multihead-self-attention", or "MLP"

    # Store intermediate results, just for debug purposes
    store_intermediate_flag = True

    # A random input tensor
    x = torch.randn(d, n_t, n_in)

    # Perform NN training for several models
    NN_result = np.zeros((N_net, num_layers, d, n_t, n))
    for N_i in range(N_net):
        stack = NN(
            n,
            n_h,
            n_t,
            n_in,
            num_layers,
            weight_input_std=weight_input_std,
            weight_E_std=weight_E_std,
            weight_Q_std=weight_Q_std,
            n_invariance_flag=n_invariance_flag,
            type=NN_type,
        )

        _ = stack(x, store_intermediate_flag=store_intermediate_flag)

        if store_intermediate_flag:
            NN_result[N_i, :, :, :, :] = stack.layer_outputs

        del stack

    # Save the results

    # Save the results in a directory with a time marking
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m%d-%H%M-%S")
    dir = f"results/{formatted_time}"
    os.makedirs(dir, exist_ok=True)

    hyperparameters = {
        "N_net": N_net,
        "d": d,
        "n_t": n_t,
        "n_in": n_in,
        "n": n,
        "n_h": n_h,
        "num_layers": num_layers,
        "weight_input_std": weight_input_std,
        "weight_E_std": weight_E_std,
        "weight_Q_std": weight_Q_std,
        "n_invariance_flag": n_invariance_flag,
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
    weight_E_std = hyperparameters["weight_E_std"]
    weight_Q_std = hyperparameters["weight_Q_std"]
    weight_input_std = hyperparameters["weight_input_std"]
    n_invariance_flag = hyperparameters["n_invariance_flag"]

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
    expected_shape=(N_net, num_layers, d, n_t, n),
):
    r"""
    Calculates the correlation function over an ensemble of neural networks.
    Assumes a specific dataset, token and neural index, i.e.
    r_1[N_i] = r_{\delta_1 t_1 i}^ell
    Note that for n or m=2 or odd, this equals the connected 'diagrams'.
    input:
    r_1: np.ndarray, shape=(N_net,layers,d,n_t,n)
    r_2: np.ndarray, shape=(N_net,layers,d,n_t,n)
    power_1: int, the power to which to raise r_1
    power_2: int, the power to which to raise r_2
    """

    # Ensure correct input
    assert r_1.shape == expected_shape

    if r_2:
        return np.mean(r_1**power_1 * r_2**power_2, axis=0)
    else:
        return np.mean(r_1**power_1, axis=0)


# Get the index or average over all indices
def get_index_or_avg(NN_result: np.ndarray, delta, t, i, d: int = d):
    r"""
    Returns the correlation function for a specific index or the average over all indices.
    input:
    NN_result: np.ndarray, shape=(layers,d,n_t,n)
    delta: int, the delta index, or 'avg'
    t: int, the t index, or 'avg'
    i: int, the i index, or 'avg'
    output:
    NN_result: np.ndarray, shape=(layers)
    """
    if i == "avg":
        NN_result = np.mean(NN_result, axis=3)
    else:
        NN_result = NN_result[:, :, :, i]

    if t == "avg":
        NN_result = np.mean(NN_result, axis=2)
    else:
        NN_result = NN_result[:, :, t]

    if delta == "avg":
        NN_result = np.mean(NN_result, axis=1)
    else:
        NN_result = NN_result[:, delta]

    return NN_result


NN_result_r1 = get_index_or_avg(correlation_function_NN(NN_result, 1), delta, t, i)
NN_result_r2 = get_index_or_avg(correlation_function_NN(NN_result, 2), delta, t, i)
NN_result_r3 = get_index_or_avg(correlation_function_NN(NN_result, 3), delta, t, i)

if NN_type == "multihead-self-attention":
    # G shape=(n_layers, d, d, t, t)
    G = G(
        num_layers,
        weight_Q_std,
        weight_E_std,
        n,
        n_h,
        weight_input_std,
        x,
        n_independent_flag=n_invariance_flag,
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
else:
    G = None

# Compare the results


def plot_comparison(
    correlation_NN_r1,
    correlation_NN_r2,
    correlation_NN_r3,
    corellation_G=None,
    figname="test.png",
    dir=dir,
    hyperparameters=hyperparameters,
):
    r"""
    Plots the comparison between the theoretical forward equation
    and neural network corellation functions per layer.
    Note that the correlation function per layer,
    e.g. set \delta=1, t=1, i=1.
    Input:
    correlation_NN: np.ndarray, shape=(layers)
    corellation_G: np.ndarray, shape=(layers), optional
    """

    fig, ax = plt.subplots(3, 2, figsize=(10, 5))
    layers = np.arange(correlation_NN_r1.shape[0])

    ax[0, 0].plot(layers, correlation_NN_r1)
    ax[0, 0].set_title("Neural Network Correlation Functions")
    ax[0, 0].set_ylabel(f"r1_{delta},{t},{i}")
    ax[0, 0].set_xticks([])

    ax[1, 0].plot(layers, correlation_NN_r2)
    ax[1, 0].set_ylabel(f"r2_{delta},{t},{i}")
    ax[1, 0].set_xticks([])

    ax[2, 0].plot(layers, correlation_NN_r3)
    ax[2, 0].set_ylabel(f"r3_{delta},{t},{i}")
    ax[2, 0].set_xlabel("Layers")

    if corellation_G is not None:
        ax[1, 1].plot(layers, corellation_G)
        ax[0, 1].set_title("Theoretical correlation function")

    ax[1, 1].set_xticks([])
    ax[1, 1].set_xticks([])
    ax[2, 1].set_xlabel("Layers")

    # Add hyperparameters as text on the side
    hyperparameters_text = "\n".join(
        [
            f"{key}: {value}"
            for key, value in hyperparameters.items()
            # (f"{key}: {value}" if len(f"{key}: {value}") < 15 else f"{key}:\n {value}")
            # for key, value in hyperparameters.items()
        ]
    )
    plt.gcf().text(
        0.84, 0.5, hyperparameters_text, fontsize=10, verticalalignment="center"
    )

    # Set the title for the entire figure
    fig.suptitle("Comparison of Correlation Functions", fontsize=16)

    plt.tight_layout(
        rect=[0, 0, 0.84, 1]
    )  # Adjust layout to make room for the text box
    plt.savefig(dir + "/" + figname)


plot_comparison(
    NN_result_r1,
    NN_result_r2,
    NN_result_r3,
    corellation_G=G,
    figname=f"{NN_type}_d{delta}-t{t}-i{i}.png",
)

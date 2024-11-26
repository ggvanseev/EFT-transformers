import torch
import numpy as np
from self_attention_nn import NN
import matplotlib.pyplot as plt
import os
from datetime import datetime
import yaml

# This file performes a comparison between the theoretical forward equation
# for the covariance matrix of the l-th layer to a neural network layer.

# Save the results in a directory with a time marking
current_time = datetime.now()
formatted_time = current_time.strftime("%m%d-%H%M-%S")
dir = f"results/{formatted_time}"
os.makedirs(dir, exist_ok=True)


# Choose hyperparameters
N_net = 20  # Number of neural networks
d = 4  # Number of samples in the batch
n_t = 10  # Number of tokens
n_in = 1  # Number of input features
n = 500  # Number of features/neurons in hidden/output layers
n_h = 1  # Number of attention heads
num_layers = 2  # Total number of layers in the stack
# Width of the Gaussian distribution for initialization
weight_input_std = 0.5
weight_E_std = 0.5
weight_Q_std = 0.5
n_invariance_flag = False

# A random input tensor
x = torch.randn(d, n_t, n_in)


# Perform NN training for several models
NN_result = np.zeros((N_net, num_layers))  # np.zeros((N_net, num_layers, d, n_t, n))
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
    )

    _ = stack(x, store_intermediate_flag=True)

    NN_result[N_i, :] = [stack.layer_outputs[l][0, 0, 0] for l in range(num_layers)]
    # NN_result[N_i, :, :, :, :] = stack.layer_outputs

    del stack


# Save the results
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
}

with open(f"{dir}/hyperparameters.yaml", "w") as file:
    yaml.dump(hyperparameters, file)

np.save(f"{dir}/NN_result.npy", NN_result)
np.save(f"{dir}/random_input_data_x.npy", x.numpy())


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
    input:
    r_1: np.ndarray, shape=(N_net,layers,d,n_t,n)
    r_2: np.ndarray, shape=(N_net,layers,d,n_t,n)
    power_1: int, the power to which to raise r_1
    power_2: int, the power to which to raise r_2
    """

    # Ensure correct input
    # assert r_1.shape == expected_shape

    if r_2:
        return np.mean(r_1**power_1 * r_2**power_2, axis=0)
    else:
        return np.mean(r_1**power_1, axis=0)


NN_result_r1 = correlation_function_NN(NN_result, 1)


# Compare the results


def plot_comparison(
    correlation_NN,
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

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    layers = np.arange(correlation_NN.shape[0])

    ax[0].plot(layers, correlation_NN)
    ax[0].set_title("NN correlation function")

    if corellation_G:
        ax[1].plot(layers, corellation_G)
        ax[1].set_title("Theoretical correlation function")

        # Add hyperparameters as text on the side
    hyperparameters_text = "\n".join(
        [
            f"{key}: {value}" if len(f"{key}: {value}") < 15 else f"{key}:\n {value}"
            for key, value in hyperparameters.items()
        ]
    )
    plt.gcf().text(
        0.88, 0.5, hyperparameters_text, fontsize=10, verticalalignment="center"
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for the text box
    plt.savefig(dir + "/" + figname)


plot_comparison(NN_result_r1)

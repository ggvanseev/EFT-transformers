import matplotlib.pyplot as plt
import numpy as np


def plot_correlation_function_comparison(
    correlation_NN_r1,
    correlation_NN_r2,
    correlation_NN_r3,
    corellation_G=None,
    figname="test.png",
    dir=None,
    hyperparameters=None,
    delta=1,
    t=1,
    i=1,
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

    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    layers = np.arange(correlation_NN_r1.shape[0])

    ax[0].plot(layers, correlation_NN_r1)
    ax[0].set_ylabel(f"r1_{delta},{t},{i}")
    ax[0].set_xticks([])

    ax[1].plot(layers, correlation_NN_r2, label="Neural Network")
    ax[1].set_ylabel(f"r2_{delta},{t},{i}")
    ax[1].set_yscale("log")
    ax[1].set_xticks([])

    ax[2].plot(layers, correlation_NN_r3)
    ax[2].set_ylabel(f"r3_{delta},{t},{i}")
    ax[2].set_xlabel("Layers")

    if corellation_G is not None:
        ax[1].plot(layers, corellation_G, label="Theoretical correlation function")

    ax[1].legend()
    # Add hyperparameters as text on the side
    hyperparameters_text = "\n".join(
        [
            (f"{key}:\n {value}" if ("flag" in str(key)) else f"{key}: {value}")
            for key, value in hyperparameters.items()
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


def f_gaussian(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_histogram_comparison(
    NN_result: np.ndarray,
    x: np.ndarray,
    var_theory: np.ndarray = None,
    figname="histogram.png",
    dir=None,
    hyperparameters=None,
    hist_bins=300,
):
    r"""
    Plots the theoretical and neural network histogram per layer.
    Input:
    NN_result: np.ndarray, shape=(layers,N_net)
    x: np.ndarray, shape=(d, n_t, n)
    var_theory: np.ndarray, shape=(layers)
    figname: str, the name of the figure
    dir: str, the directory to save the figure
    hyperparameters: dict, the hyperparameters of the run
    hist_bins: int, the number of bins for the histogram
    """
    num_layers = NN_result.shape[0]

    # Setup the figure to be a square grid
    n_plots_per_side = int(np.sqrt(num_layers))
    if np.sqrt(num_layers) % n_plots_per_side != 0:
        n_plots_per_side += 1
    fig, axs = plt.subplots(
        n_plots_per_side,
        n_plots_per_side,
        figsize=(7 * n_plots_per_side, 7 * n_plots_per_side),
    )
    axs = axs.ravel()

    # Loop over the layers
    for l, ax in enumerate(axs):
        if l - 1 >= num_layers:
            ax.axis("off")
            continue
        elif l >= num_layers:
            # ax.hist(x[0], bins=hist_bins, density=True, label="Input for batch 0")
            # ax.legend(fontsize=13)
            # ax.set_ylabel("Probability Density")
            # ax.set_xlabel("Layers")
            ax.axis("off")
            continue

        x_grid = np.linspace(
            min(NN_result[l]),
            max(NN_result[l]),
            1000,
        )

        # Plot the leading order theoretical distribution, which is a Gaussian.
        if var_theory is not None:
            theoretical_sigma = np.sqrt(var_theory[l])
            theoretical_gaussian = f_gaussian(x_grid, sigma=theoretical_sigma)
            ax.plot(
                x_grid,
                theoretical_gaussian,
                linestyle="--",
                color="crimson",
                markersize=0,
                label=rf"LO Distribution layer {l+1}: mean=0 $\pm${theoretical_sigma:.5f}",
                linewidth=3.5,
            )
            ax.set_ylim(top=np.max(theoretical_gaussian) * 1.3)

        # Hisogram the numerical results
        ax.hist(
            NN_result[l],
            bins=hist_bins,
            density=True,
            label=rf"NN layer {l+1}: mean={np.mean(NN_result[l]):.5f}$\pm${np.std(NN_result[l]):.5f}",
        )

        ax.legend(fontsize=13)
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("Bin values")

    # Add hyperparameters as text on the side
    hyperparameters_text = "\n".join(
        [
            (f"{key}:\n {value}" if ("flag" in str(key)) else f"{key}: {value}")
            for key, value in hyperparameters.items()
        ]
    )
    plt.gcf().text(
        0.84,
        0.5,
        hyperparameters_text,
        fontsize=4 * n_plots_per_side + 10,
        verticalalignment="center",
    )

    # Set the title for the entire figure
    fig.suptitle("Comparison of Correlation Functions", fontsize=16)

    plt.tight_layout(
        rect=[0, 0, 0.84, 1]
    )  # Adjust layout to make room for the text box
    plt.savefig(dir + "/" + figname)

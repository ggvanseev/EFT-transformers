import torch
import numpy as np

# This is all for a linear activation function.
# Only at initialisation.

# Assume that each layer is Gaussian with a covariance matrix G
# Note that G = G_{\delta_1\delta_2 t_1t_2 i j} = G_{\delta_1\delta_2 t_1t_2} \delta_{ij}


# For the first layer
def G_MHSA_1(c_w: float, x: np.ndarray, n_independent_flag: bool = False) -> np.ndarray:
    r"""
    Latex notation definition of G^{(1)} for the first layer:
    G^{(1)}=\frac{c_W}{2}G^{(1)}_{\delta_1\delta_2t_1t_2ij}
    =\frac{c_W}{2}\delta_{ij}\sum_kx_{\delta_1t_1k}x_{\delta_2t_2k}
    input:
    c_w: float, the weight of the first layer
    x: np.ndarray, shape=(d, t, i), the input tensor with n_in features
    n_independent_flag: bool, whether to divide by the number of features
    output:
    G_1: np.ndarray, G_{\delta_1\delta_2 t_1t_2}
    shape=(d, d, t, t), the covariance matrix of the first layer
    """
    n_in = x.shape[2]

    if n_independent_flag:
        c_w /= n_in

    return c_w / 2 * np.einsum("dti,eui->detu", x, x)


# For the l-th layer, l>1
# Note that you have to do something about the sums over the neural indices.
# Instead of performing those, since they are just sums, you can just multiply
# the result by the number of neurons.


def G_MHSA_forward(
    G_prime: np.ndarray,
    c_q: float,
    c_e: float,
    n: int,
    n_h: int,
    invariance_flags={"n": True, "n_t": False, "n_h": False},
) -> np.ndarray:
    r"""
    Latex notation definition of G for the forward equation:
    G=G_{\delta_1\delta_2t_1t_2ij}=2c_qc_e\delta_{ij}n^3n_h
    G'_{\delta_1\delta_2t_1t_2ll}\sum_{t_3t_4}\Theta(t_1-t_3)\Theta(t_2-t_4)
    G'_{\delta_1\delta_2t_3t_4kk}G'_{\delta_1\delta_2t_3t_4mm}
    input:
    G_prime: np.ndarray, shape=(d, d, t, t),
    the covariance matrix of the previous layer
    c_q: float, the weight of the query tensor
    c_e: float, the weight of the encoder-decoder tensor
    n: int, the number of neurons
    n_h: int, the number of heads
    invariance_flags:dict{bool}, whether to divide by the number of features
    """
    factor = 1
    n_t = G_prime.shape[2]

    if not invariance_flags["n"]:
        factor *= n**3
    if invariance_flags["n_t"]:
        factor /= n_t**2
    if invariance_flags["n_h"]:
        factor /= n_h

    # Theta
    Theta = np.tril(np.ones((n_t, n_t)))

    return (
        2
        * c_q
        * c_e
        * factor
        * np.einsum(
            "abtu,tv,uw,abvw,abvw->abtu", G_prime, Theta, Theta, G_prime, G_prime
        )
    )


def G_MHSA(
    n_layers: int,
    Q_std: float,
    E_std: float,
    n: int,
    n_h: int,
    c_w: float,
    x: np.ndarray,
    invariance_flags={"n": True, "n_t": False, "n_h": False},
) -> np.ndarray:
    """
    Use the forward equation to calculate up to the n_layers-th layer

    input:
    n_layers: int, the number of layers
    c_q: float, the standard deviation of the query weight
    c_e: float, the standard deviation of the encoder-decoder weight
    n: int, the number of neurons
    n_h: int, the number of heads
    c_w: float, the weight of the first layer
    x: np.ndarray, shape=(d, t, i), the input tensor with n_in features
    invariance_flags: dict{bool}, whether to divide by the number of features
    output:
    G_all: np.ndarray, shape=(n_layers, d, d, t, t),
    the covariance matrices of all the layers
    """
    d = x.shape[0]  # Number of datset samples
    n_t = x.shape[1]  # Number of tokens

    # Convert standard deviation to covariance
    c_q = Q_std**2
    c_e = E_std**2

    # A matrix to store all the layers
    G_all = np.zeros((n_layers, d, d, n_t, n_t))

    for layer in range(n_layers):
        if layer == 0:
            G_all[layer] = G_MHSA_1(c_w, x, invariance_flags["n"])
        else:
            G_all[layer] = G_MHSA_forward(
                G_all[layer - 1], c_q, c_e, n, n_h, invariance_flags
            )

    return G_all


# For the linear MLP
def G_MLP_1(c_w: float, x: np.ndarray, n_independent_flag: bool = False) -> np.ndarray:
    r"""
    Latex notation definition of G^{(1)} for the first layer:
    G^{(1)}=\frac{c_W}{2}G^{(1)}_{\delta_1\delta_2ij}
    =\frac{c_W}{2}\delta_{ij}\sum_kx_{\delta_1k}x_{\delta_2k}
    input:
    c_w: float, the weight of the first layer
    x: np.ndarray, shape=(d,i), the input tensor with n_in features
    n_independent_flag: bool, whether to divide by the number of features
    output:
    G_1: np.ndarray, G_{\delta_1\delta_2}
    shape=(d, d), the covariance matrix of the first layer
    """
    n_in = x.shape[1]

    if n_independent_flag:
        c_w /= n_in

    return c_w / 2 * np.einsum("di,ei->de", x, x)


def G_MLP_forward(
    G_prime: np.ndarray,
    c_w: float,
    n: int,
    n_independent_flag: bool = False,
) -> np.ndarray:
    r"""
    Latex notation definition of G for the forward equation:
    G=G_{\delta_1\delta_2ij}=\frac{C_w}{2}\delta_{ij}\sum_kG'_{\delta_1\delta_2kk}
    =\frac{C_wn}{2}\delta_{ij}G'_{\delta_1\delta_2}
    input:
    G_prime: np.ndarray, shape=(d, d),
    the covariance matrix of the previous layer
    c_w: float, the covariance of the weight
    n: int, the number of neurons
    n_independent_flag: bool, whether to divide by the number of features
    """

    if n_independent_flag:

        return c_w / 2 * G_prime
    else:
        return c_w * n / 2 * G_prime


def G_MLP(
    n_layers: int,
    n: int,
    W_std: float,
    x: np.ndarray,
    n_independent_flag: bool = False,
) -> np.ndarray:
    """
    Use the forward equation to calculate up to the n_layers-th layer

    input:
    n_layers: int, the number of layers
    n: int, the number of neurons
    n_h: int, the number of heads
    W_std: float, the standard deviation of the weights
    x: np.ndarray, shape=(d, t, i), the input tensor with n_in features
    n_independent_flag: bool, whether to divide by the number of features
    output:
    G_all: np.ndarray, shape=(n_layers, d, d),
    the covariance matrices of all the layers
    """
    d = x.shape[0]  # Number of datset samples

    # Convert standard deviation to covariance
    c_w = W_std**2

    # A matrix to store all the layers
    G_all = np.zeros((n_layers, d, d))

    for layer in range(n_layers):
        if layer == 0:
            G_all[layer] = G_MLP_1(c_w, x, n_independent_flag)
        else:
            G_all[layer] = G_MLP_forward(G_all[layer - 1], c_w, n, n_independent_flag)

    return G_all


if __name__ == "__main__":
    # Example usage:
    d = 4  # Number of samples in the batch
    n_t = 10  # Number of tokens
    n_in = 1  # Number of input features
    n = 16  # Number of features/neurons in hidden/output layers
    n_h = 8  # Number of attention heads
    num_layers = 3  # Total number of layers in the stack
    # Width of the Gaussian distribution for initialization
    W_std = 0.1
    E_std = 0.1
    Q_std = 0.1

    invariance_flags = {"n": True, "n_t": False, "n_h": False}

    NN_type = "MLP"  # "MHSA", or "MLP"

    x = torch.randn(d, n_t, n_in)  # Input tensor with size n_in per token
    x = x.cpu().numpy()

    if NN_type == "MHSA":
        output = G_MHSA(
            num_layers,
            Q_std,
            E_std,
            n,
            n_h,
            W_std,
            x,
            invariance_flags=invariance_flags,
        )
    elif NN_type == "MLP":
        output = G_MLP(
            num_layers,
            n,
            W_std,
            x[:, 0, :],
            n_independent_flag=True,
        )

    print("Output shape:", output.shape)  # Should be (d, n_t, n)

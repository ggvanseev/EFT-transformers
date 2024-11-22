import torch

# This is all for a linear activation function.
# Only at initialisation.

# Assume that each layer is Gaussian with a covariance matrix G
# Note that G = G_{\delta_1\delta_2 t_1t_2 i j} = G_{\delta_1\delta_2 t_1t_2} \delta_{ij}


# For the first layer
def G_1(c_w: float, x: torch.tensor, n_independent_flag: bool = False) -> torch.tensor:
    r"""
    Latex notation definition of G^{(1)} for the first layer:
    G^{(1)}=\frac{c_W}{2}G^{(1)}_{\delta_1\delta_2t_1t_2ij}
    =\frac{c_W}{2}\delta_{ij}\sum_kx_{\delta_1t_1k}x_{\delta_2t_2k}
    input:
    c_w: float, the weight of the first layer
    x: torch.tensor, shape=(d, t, i), the input tensor with n_in features
    n_independent_flag: bool, whether to divide by the number of features
    output:
    G_1: torch.tensor, G_{\delta_1\delta_2 t_1t_2}
    shape=(d, d, t, t), the covariance matrix of the first layer
    """
    n_in = x.shape[2]

    if n_independent_flag:
        c_w /= n_in

    return c_w / 2 * torch.einsum("dti,eui->detu", x, x)


# For the l-th layer, l>1
# Note that you have to do something about the sums over the neural indices.
# Instead of performing those, since they are just sums, you can just multiply
# the result by the number of neurons.


def G_forward(
    G_prime: torch.tensor,
    c_q: float,
    c_e: float,
    n: int,
    n_h: int,
    n_independent_flag: bool = False,
) -> torch.tensor:
    r"""
    Latex notation definition of G for the forward equation:
    G=G_{\delta_1\delta_2t_1t_2ij}=2c_qc_e\delta_{ij}n^3n_h
    G'_{\delta_1\delta_2t_1t_2ll}\sum_{t_3t_4}
    G'_{\delta_1\delta_2t_3t_4kk}G'_{\delta_1\delta_2t_3t_4mm}
    input:
    G_prime: torch.tensor, shape=(d, d, t, t),
    the covariance matrix of the previous layer
    c_q: float, the weight of the query tensor
    c_e: float, the weight of the encoder-decoder tensor
    n: int, the number of neurons
    n_h: int, the number of heads
    n_independent_flag: bool, whether to divide by the number of features
    """

    if n_independent_flag:
        n_t = G_prime.shape[2]

        return (
            2
            * c_q
            * c_e
            / n_t**2
            * torch.einsum("abtu,abvw,abvw->abtu", G_prime, G_prime, G_prime)
        )
    else:
        return (
            2
            * c_q
            * c_e
            * n**3
            * n_h
            * torch.einsum("abtu,abvw,abvw->abtu", G_prime, G_prime, G_prime)
        )


def G(
    n_layers: int,
    c_q: float,
    c_e: float,
    n: int,
    n_h: int,
    c_w: float,
    x: torch.tensor,
    n_independent_flag: bool = False,
) -> torch.tensor:
    """
    Use the forward equation to calculate up to the n_layers-th layer

    input:
    n_layers: int, the number of layers
    c_q: float, the weight of the query tensor
    c_e: float, the weight of the encoder-decoder tensor
    n: int, the number of neurons
    n_h: int, the number of heads
    c_w: float, the weight of the first layer
    x: torch.tensor, shape=(d, t, i), the input tensor with n_in features
    n_independent_flag: bool, whether to divide by the number of features
    output:
    G_all: torch.tensor, shape=(n_layers, d, d, t, t),
    the covariance matrices of all the layers
    """
    d = x.shape[0]  # Number of datset samples
    n_t = x.shape[1]  # Number of tokens

    # A matrix to store all the layers
    G_all = torch.zeros((n_layers, d, d, n_t, n_t))

    for layer in range(n_layers):
        if layer == 0:
            G_all[layer] = G_1(c_w, x, n_independent_flag)
        else:
            G_all[layer] = G_forward(
                G_all[layer - 1], c_q, c_e, n, n_h, n_independent_flag
            )

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
    weight_input_std = 0.1
    weight_E_std = 0.1
    weight_Q_std = 0.1

    x = torch.randn(d, n_t, n_in)  # Input tensor with size n_in per token

    output = G(
        num_layers,
        weight_Q_std,
        weight_E_std,
        n,
        n_h,
        weight_input_std,
        x,
        n_independent_flag=True,
    )

    print("Output shape:", output.shape)  # Should be (d, n_t, n)

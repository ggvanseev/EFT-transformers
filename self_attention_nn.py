import torch
import torch.nn as nn
import numpy as np


class LinearMLPBlock(nn.Module):
    def __init__(self, n_in, n, N_net: int = 1, W_std=0.02, n_invariance_flag=False):
        """
        Initialize a single layer.
        :param n_in: Number of input features
        :param n: Number of output features
        :param N_net: Number of networks in the ensemble
        :param W_std: Standard deviation of Gaussian distribution for weight initialization
        :param n_invariance_flag: Flag to enable weight invariance
        """
        super(LinearMLPBlock, self).__init__()
        self.n = n
        self.n_in = n_in

        self.W = nn.Parameter(torch.empty(N_net, n, n_in))  # Shape (n, n_in)

        self._initialize_weights(W_std=W_std, n_invariance_flag=n_invariance_flag)

    def _initialize_weights(self, W_std, n_invariance_flag=False):
        """
        Custom initialization of weights with Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution
        """
        if n_invariance_flag:
            W_std /= np.sqrt(self.n_in)

        nn.init.normal_(
            self.W, mean=0.0, std=W_std
        )  # Initialize input transform with Gaussian

    def forward(self, x):
        """
        Forward pass of the layer.
        :param c: Input tensor of shape: (N_net,d, n_t, n_in)
        :return: Output tensor of shape (N_net, d, n_t, n)
        """
        return torch.einsum("Nji,Ndti->Ndtj", self.W, x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n,
        n_h,
        n_t,
        n_in=None,
        N_net=1,
        E_std=0.02,
        Q_std=0.02,
        invariance_flags={"n": False, "n_t": False, "n_h": False},
    ):
        """
        Initialize a single layer.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param n_in: Number of input features, None if the input is the output of the previous layer
        :param N_net: Number of networks in the ensemble
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        """
        super(AttentionBlock, self).__init__()
        self.n = n
        self.n_h = n_h
        self.n_t = n_t
        if n_in is None:
            n_in = n
        self.n_in = n_in
        self.N_net = N_net

        # Learnable weights for feature mixing
        self.E = nn.Parameter(
            torch.empty(N_net, n_h, n, n_in)
        )  # Shape (N_net, n_h, n, n)

        # Learnable weights for attention computation
        self.Q = nn.Parameter(
            torch.empty(N_net, n_h, n_in, n_in)
        )  # Shape (N_net, n_h, n, n)

        # Initialize weights with specified Gaussian width
        self._initialize_weights(
            E_std=E_std,
            Q_std=Q_std,
            n_invariance_flag=invariance_flags["n"],
            nt_invariance_flag=invariance_flags["n_t"],
            nh_invariance_flag=invariance_flags["n_h"],
        )

    def _initialize_weights(
        self,
        E_std,
        Q_std,
        n_invariance_flag=False,
        nt_invariance_flag=False,
        nh_invariance_flag=False,
    ):
        """
        Custom initialization of weights with Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution
        """
        if n_invariance_flag:
            E_std /= np.sqrt(self.n_in)
            Q_std /= self.n_in  # equals np.sqrt(self.n_in**2)
        if nt_invariance_flag:
            E_std /= self.n_t  # equals np.sqrt(self.n_t**2)
        if nh_invariance_flag:
            E_std /= np.sqrt(self.n_h)

        nn.init.normal_(self.E, mean=0.0, std=E_std)  # Initialize E with Gaussian
        nn.init.normal_(self.Q, mean=0.0, std=Q_std)  # Initialize Q with Gaussian

    def forward(self, r_prime):
        """
        Forward pass of the layer.
        :param r_prime: Input tensor of shape: (N_net, d, n_t, n_in)
        :return: Output tensor of shape (N_net, d, n_t, n)
        """
        # Compute attention scores Omega_{\delta t_1t_2}^h
        # Omega = [\delta, h, t_1, t_2]
        # Shape of Omega: (N_net, d, n_h, n_t, n_t)
        Omega = torch.zeros(
            (self.N_net, r_prime.size(1), self.n_h, self.n_t, self.n_t),
            device=r_prime.device,
        )

        # Compute Omega_{\delta t_1t_2}^h = r'_{\delta t_1 i} Q^h_{ij} r'_{\delta t_2 j}
        Omega = torch.einsum("Nbti,Nhij,Nbuj->Nbhtu", r_prime, self.Q, r_prime)

        # Apply Omega_{\delta t_1t_2}^h = \Theta(t_1-t_2)omega_{\delta t_1t_2}^h
        # Create a lower triangular mask with shape [t_1,t_2]
        Theta = torch.tril(torch.ones(self.n_t, self.n_t, device=r_prime.device))

        # Shape of r: (N_net, d, n_t, n)
        # Compute r_{\delta,t_1,i} = \Omega_{\delta t_1t_2}^h\Theta(t_1-t_2)
        # E^h_{ij} r'_{\delta  t_2j}
        # per network N_i
        r = torch.einsum("Nbhtu,tu,Nhij,Nbuj->Nbti", Omega, Theta, self.E, r_prime)

        return r


class NN(nn.Module):
    def __init__(
        self,
        n,
        n_h,
        n_t,
        n_in,
        num_layers,
        N_net=1,
        W_std=0.02,
        E_std=0.02,
        Q_std=0.02,
        invariance_flags={"n": False, "n_t": False, "n_h": False},
        NN_type="MHSA",
    ):
        """
        Initialize a stack of layers for N_net networks.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param num_layers: Total number of layers in the stack
        :param N_net: Number of networks in the ensemble
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        :param invariance_flags: Flag to enable weight invariance
        :param NN_type: Type of the model options: ["MHSA", "MLP"]
        """
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        n_invariance_flag = invariance_flags["n"]

        # First layer: input size 1 -> n
        if NN_type == "MHSA":
            # self.layers.append(
            #     AttentionBlock(
            #         n,
            #         n_h,
            #         n_t,
            #         n_in,
            #         N_net=N_net,
            #         E_std=W_std,
            #         Q_std=Q_std,
            #         n_invariance_flag=n_invariance_flag,
            #     )
            # )
            self.layers.append(
                LinearMLPBlock(
                    n_in,
                    n,
                    N_net=N_net,
                    W_std=W_std,
                    n_invariance_flag=n_invariance_flag,
                )
            )
        elif NN_type == "MLP":
            self.layers.append(
                LinearMLPBlock(
                    n_in,
                    n,
                    N_net=N_net,
                    W_std=W_std,
                    n_invariance_flag=n_invariance_flag,
                )
            )

        # Subsequent layers: input and output size n
        for _ in range(num_layers - 1):
            if NN_type == "MHSA":
                self.layers.append(
                    AttentionBlock(
                        n,
                        n_h,
                        n_t,
                        N_net=N_net,
                        E_std=E_std,
                        Q_std=Q_std,
                        invariance_flags=invariance_flags,
                    )
                )
            elif NN_type == "MLP":
                self.layers.append(
                    LinearMLPBlock(
                        n,
                        n,
                        N_net=N_net,
                        W_std=E_std,
                        n_invariance_flag=n_invariance_flag,
                    )
                )

        # Store the output of each layer
        # This is done in numpy
        self.layer_outputs = []

    def forward(self, s, store_intermediate_flag: bool = False):
        """
        Forward pass through the stack of layers.
        :param s: Input tensor of shape (d, n_t, 1)
        :return: Output tensor of shape (d, n_t, n)
        """
        for layer in self.layers:
            s = layer(s)  # Pass output of one layer as input to the next
            if store_intermediate_flag:
                self.layer_outputs.append(s.detach().clone().cpu().numpy())
        return s


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

    # Number of networks in the ensemble
    N_net = 2
    N_type = "MLP"  # "MHSA", or "MLP"

    x = torch.stack(
        [torch.randn(d, n_t, n_in)] * N_net
    )  # Input tensor with size n_in per token

    stack = NN(
        n,
        n_h,
        n_t,
        n_in,
        num_layers,
        N_net=N_net,
        W_std=W_std,
        E_std=E_std,
        Q_std=Q_std,
        invariance_flags=invariance_flags,
        NN_type=N_type,
    )
    output = stack(x, store_intermediate_flag=True)
    intermediate_outputs = stack.layer_outputs

    print("Output shape:", output.shape)  # Should be (d, n_t, n)

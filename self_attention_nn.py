import torch
import torch.nn as nn


class LinearMLPBlock(nn.Module):
    def __init__(self, n_in, n, std_W=0.02, n_invariance_flag=False):
        super(LinearMLPBlock, self).__init__()
        self.n = n
        self.n_in = n_in

        self.W = nn.Parameter(torch.empty(n, n_in))  # Shape (n, n_in)

        self._initialize_weights(std_W=std_W, n_invariance_flag=n_invariance_flag)

    def _initialize_weights(self, std_W, n_invariance_flag=False):
        """
        Custom initialization of weights with Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution
        """
        if n_invariance_flag:
            std_W /= self.n_in

        nn.init.normal_(
            self.W, mean=0.0, std=std_W
        )  # Initialize input transform with Gaussian

    def forward(self, x):
        """
        Forward pass of the layer.
        :param c: Input tensor of shape: (d, n_t, n_in)
        :return: Output tensor of shape (d, n_t, n)
        """
        return torch.einsum("dti,ji->dtj", x, self.W)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n,
        n_h,
        n_t,
        n_in=None,
        weight_E_std=0.02,
        weight_Q_std=0.02,
        n_invariance_flag=False,
    ):
        """
        Initialize a single layer.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param n_in: Number of input features, None if the input is the output of the previous layer
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        """
        super(AttentionBlock, self).__init__()
        self.n = n
        self.n_h = n_h
        self.n_t = n_t
        if n_in is None:
            n_in = n
        self.n_in = n_in

        # Learnable weights for feature mixing
        self.E = nn.Parameter(torch.empty(n_h, n, n_in))  # Shape (n_h, n, n)

        # Learnable weights for attention computation
        self.Q = nn.Parameter(torch.empty(n_h, n_in, n_in))  # Shape (n_h, n, n)

        # Initialize weights with specified Gaussian width
        self._initialize_weights(
            std_E=weight_E_std, std_Q=weight_Q_std, n_invariance_flag=n_invariance_flag
        )

    def _initialize_weights(self, std_E, std_Q, n_invariance_flag=False):
        """
        Custom initialization of weights with Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution
        """
        if n_invariance_flag:
            std_E /= self.n_in * self.n_h**2 * self.n_t**2
            std_Q = std_Q * self.n_h / self.n_in**2

        nn.init.normal_(self.E, mean=0.0, std=std_E)  # Initialize E with Gaussian
        nn.init.normal_(self.Q, mean=0.0, std=std_Q)  # Initialize Q with Gaussian

    def forward(self, r_prime):
        """
        Forward pass of the layer.
        :param r_prime: Input tensor of shape: (d, n_t, n_in)
        :return: Output tensor of shape (d, n_t, n)
        """
        # Compute attention scores Omega_{\delta t_1t_2}^h
        # Omega = [\delta, h, t_1, t_2]
        # Shape of Omega: (d, n_h, n_t, n_t)
        Omega = torch.zeros(
            (r_prime.size(0), self.n_h, self.n_t, self.n_t), device=r_prime.device
        )

        # Compute Omega_{\delta t_1t_2}^h = r'_{\delta t_1 i} Q^h_{ij} r'_{\delta t_2 j}
        Omega = torch.einsum("bti,hij,buj->bhtu", r_prime, self.Q, r_prime)

        # Apply Omega_{\delta t_1t_2}^h = \Theta(t_1-t_2)omega_{\delta t_1t_2}^h
        # Create a lower triangular mask in the [t_1,t_2] matrix and apply it to
        # each element of the batch and each head.
        mask = (
            torch.tril(torch.ones(self.n_t, self.n_t, device=r_prime.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        Omega = Omega * mask

        # Note that the above computation is equivalent to:
        # Omega1 = torch.zeros_like(Omega)
        # for b in range(Omega.size(0)):
        #     for h in range(Omega.size(1)):
        #         Omega1[b, h, :, :] = torch.tril(Omega[b, h, :, :])

        # Compute the final output r_{\delta,t,i}
        # r = [\delta, t, i]
        # Shape of r: (d, n_t, n)
        r = torch.zeros_like(r_prime)

        # Compute r_{\delta,t_1,i} = \Omega_{\delta t_1t_2}^h E^h_{ij} r'_{\delta  t_2j}
        r = torch.einsum("bhtu,hij,buj->bti", Omega, self.E, r_prime)

        return r


class NN(nn.Module):
    def __init__(
        self,
        n,
        n_h,
        n_t,
        n_in,
        num_layers,
        weight_input_std=0.02,
        weight_E_std=0.02,
        weight_Q_std=0.02,
        n_invariance_flag=False,
        type="multihead-self-attention",
    ):
        """
        Initialize a stack of layers.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param num_layers: Total number of layers in the stack
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        :param n_invariance_flag: Flag to enable weight invariance
        :param type: Type of the model options: ["multihead-self-attention", "MLP"]
        """
        super(NN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer: input size 1 -> n
        if type == "multihead-self-attention":
            self.layers.append(
                AttentionBlock(
                    n,
                    n_h,
                    n_t,
                    n_in,
                    weight_E_std=weight_input_std,
                    weight_Q_std=weight_Q_std,
                    n_invariance_flag=n_invariance_flag,
                )
            )
        elif type == "MLP":
            self.layers.append(
                LinearMLPBlock(
                    n_in, n, std_W=weight_input_std, n_invariance_flag=n_invariance_flag
                )
            )

        # Subsequent layers: input and output size n
        for _ in range(num_layers - 1):
            if type == "multihead-self-attention":
                self.layers.append(
                    AttentionBlock(
                        n,
                        n_h,
                        n_t,
                        weight_E_std=weight_E_std,
                        weight_Q_std=weight_Q_std,
                        n_invariance_flag=n_invariance_flag,
                    )
                )
            elif type == "MLP":
                self.layers.append(
                    LinearMLPBlock(
                        n, n, std_W=weight_E_std, n_invariance_flag=n_invariance_flag
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
    weight_input_std = 0.1
    weight_E_std = 0.1
    weight_Q_std = 0.1
    n_invariance_flag = True

    x = torch.randn(d, n_t, n_in)  # Input tensor with size n_in per token

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
    output = stack(x, store_intermediate_flag=True)
    intermediate_outputs = stack.layer_outputs

    print("Output shape:", output.shape)  # Should be (d, n_t, n)

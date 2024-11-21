import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n,
        n_h,
        n_t,
        n_in=1,
        is_first_layer=False,
        weight_input_std=0.02,
        weight_E_std=0.02,
        weight_Q_std=0.02,
    ):
        """
        Initialize a single layer.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param is_first_layer: If True, expects input size 1 and maps to n
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        """
        super(AttentionBlock, self).__init__()
        self.n = n
        self.n_h = n_h
        self.n_t = n_t
        self.n_in = n_in
        self.is_first_layer = is_first_layer

        if self.is_first_layer:
            # Input transformation only for the first layer
            self.input_transform = nn.Linear(n_in, n)

        # Learnable weights for feature mixing
        self.E = nn.Parameter(torch.empty(n_h, n, n))  # Shape (n_h, n, n)

        # Learnable weights for attention computation
        self.Q = nn.Parameter(torch.empty(n_h, n, n))  # Shape (n_h, n, n)

        # Initialize weights with specified Gaussian width
        self._initialize_weights(
            std_input=weight_input_std, std_E=weight_E_std, std_Q=weight_Q_std
        )

    def _initialize_weights(self, std_E, std_Q, std_input):
        """
        Custom initialization of weights with Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution
        """
        if self.is_first_layer:
            nn.init.normal_(
                self.input_transform.weight, mean=0.0, std=std_input
            )  # Initialize input transform with Gaussian
        nn.init.normal_(self.E, mean=0.0, std=std_E)  # Initialize E with Gaussian
        nn.init.normal_(self.Q, mean=0.0, std=std_Q)  # Initialize Q with Gaussian

    def forward(self, s):
        """
        Forward pass of the layer.
        :param s: Input tensor of shape:
                  - (d, n_t, n_in) for the first layer
                  - (d, n_t, n) for subsequent layers
        :return: Output tensor of shape (d, n_t, n)
        """
        if self.is_first_layer:
            # Transform input from size n_in to size n for the first layer
            s = self.input_transform(s)  # Shape (d, n_t, n)

        # Compute r' (feature projections for tokens): Shape (d, n_t, n)
        r_prime = s

        # Compute attention scores Omega_{\delta t_1t_2}^h
        # Omega = [\delta, h, t_1, t_2]
        # Shape of Omega: (d, n_h, n_t, n_t)
        Omega = torch.zeros((s.size(0), self.n_h, self.n_t, self.n_t), device=s.device)

        # Compute Omega_{\delta t_1t_2}^h = r'_{\delta t_1 i} Q^h_{ij} r'_{\delta t_2 j}
        Omega = torch.einsum("bti,hij,buj->bhtu", r_prime, self.Q, r_prime)

        # Apply Omega_{\delta t_1t_2}^h = \Theta(t_1-t_2)omega_{\delta t_1t_2}^h
        # Create a lower triangular mask in the [t_1,t_2] matrix and apply it to
        # each element of the batch and each head.
        mask = (
            torch.tril(torch.ones(self.n_t, self.n_t, device=s.device))
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
        r = torch.zeros_like(s)

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
    ):
        """
        Initialize a stack of layers.
        :param n: Number of neurons/features in hidden and output layers
        :param n_h: Number of attention heads
        :param n_t: Number of tokens
        :param num_layers: Total number of layers in the stack
        :param weight_std: Standard deviation of Gaussian distribution for weight initialization
        """
        super(NN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer: input size 1 -> n
        self.layers.append(
            AttentionBlock(
                n,
                n_h,
                n_t,
                n_in=n_in,
                is_first_layer=True,
                weight_input_std=weight_input_std,
                weight_E_std=weight_E_std,
                weight_Q_std=weight_Q_std,
            )
        )

        # Subsequent layers: input and output size n
        for _ in range(num_layers - 1):
            self.layers.append(
                AttentionBlock(
                    n,
                    n_h,
                    n_t,
                    is_first_layer=False,
                    weight_input_std=weight_input_std,
                    weight_E_std=weight_E_std,
                    weight_Q_std=weight_Q_std,
                )
            )

    def forward(self, s):
        """
        Forward pass through the stack of layers.
        :param s: Input tensor of shape (d, n_t, 1)
        :return: Output tensor of shape (d, n_t, n)
        """
        for layer in self.layers:
            s = layer(s)  # Pass output of one layer as input to the next
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

    s = torch.randn(d, n_t, n_in)  # Input tensor with size n_in per token

    stack = NN(
        n,
        n_h,
        n_t,
        n_in,
        num_layers,
        weight_input_std=weight_input_std,
        weight_E_std=weight_E_std,
        weight_Q_std=weight_Q_std,
    )
    output = stack(s)

    print("Output shape:", output.shape)  # Should be (d, n_t, n)

import torch.nn as nn
import torch
import numpy as np


class TokenLearnerMLP(nn.Module):
    """
    MLP TokenLearner module (https://arxiv.org/abs/2106.11297).
    """
    def __init__(self, input_shape, S=8, hidden_dims=[64], activation=nn.GELU, dropout=0.1):
        """
        Args: 
            input_shape (array): input shape excluding batch dim. Expected to follow [C, H, W].

            S (int): number of output tokens

            hidden_dims (array): MLP hidden dims

            activation (nn.Module): MLP activation

            dropout (float): MLP dropout probability
        """
        super(TokenLearnerMLP, self).__init__()

        self.input_shape = input_shape
        self.S = S
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._dropout = dropout

        self.create_layers()

    def create_layers(self):
        C, H, W = self.input_shape
        self.layer_norm = nn.LayerNorm((H, W))
        layers = []
        prev_dim = C
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation(), nn.Dropout(self._dropout)])
            prev_dim = hidden_dim
        layers.extend([nn.Linear(prev_dim, self.S)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through convolutional TokenLearner.

        Args:
            input (tensor): data of shape [B, C, H, W]

        Returns: output data (tensor) of shape [B, C, S].
        """
        B, _, H, W = input.shape
        input = self.layer_norm(input)
        input = input.permute(0, 2, 3, 1).contiguous()
        fmap = self.mlp(input)
        # spatial softmax
        fmap = torch.exp(fmap.view(B, -1, self.S))
        fmap = fmap/torch.sum(fmap, dim=1, keepdim=True)
        fmap = fmap.view(B, H, W, self.S)
        # spatial attention
        return torch.einsum("bhwc,bhws->bsc", input, fmap)


class TokenLearnerConv(nn.Module):
    """
    Convolutional TokenLearner module (https://arxiv.org/abs/2106.11297).
    """
    def __init__(self, input_shape, S=8, n_layers=4, kernel_size=3, activation=nn.GELU, dropout=0.1):
        """
        Args: 
            input_shape (array): input shape excluding batch dim. Expected to follow [C, H, W].

            S (int): number of output tokens

            n_layers (int): number of convolutional layers

            kernel_size (int): size of convolving kernel

            activation (nn.Module): CNN activation

            dropout (float): CNN dropout probability
        """
        super(TokenLearnerConv, self).__init__()

        self.input_shape = input_shape
        self.S = S
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self._dropout = dropout

        self.create_layers()

    def create_layers(self):
        C, H, W = self.input_shape
        self.layer_norm = nn.LayerNorm((H, W))
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Conv2d(C, self.S, self.kernel_size, padding="same"))
            else:
                layers.append(nn.Conv2d(self.S, self.S, self.kernel_size, padding="same"))
            if i < self.n_layers-1:
                layers.append(self.activation())
                layers.append(nn.Dropout(self._dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through convolutional TokenLearner.

        Args:
            input (tensor): data of shape [B, C, H, W]

        Returns: output data (tensor) of shape [B, S, C].
        """
        B, _, H, W = input.shape
        input = self.layer_norm(input)
        fmap = self.conv(input)
        # spatial softmax
        fmap = torch.exp(fmap.view(B, self.S, -1))
        fmap = fmap/torch.sum(fmap, dim=-1, keepdim=True)
        fmap = fmap.view(B, self.S, H, W)
        # spatial attention
        return torch.einsum("bchw,bshw->bsc", input, fmap)
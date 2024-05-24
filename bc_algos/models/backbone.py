import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class Backbone(ABC, nn.Module):
    """
    Abstract class for policy backbone. Subclass to implement different 
    backbone algorithms. 
    """
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): input embedding dim
        """
        super(Backbone, self).__init__()

        self.embed_dim = embed_dim

    @classmethod
    def factory(cls, config, embed_dim):
        """
        Create a Backbone instance from config.

        Args:
            config (addict): config object

            embed_dim (int): input embedding dim

        Returns: Backbone instance.
        """
        return cls(
            embed_dim=embed_dim,
            **config.policy.kwargs.backbone
        )

    @property
    @abstractmethod
    def output_dim(self):
        return NotImplementedError


class MLP(Backbone):
    """
    MLP policy backbone.
    """
    def __init__(self, embed_dim, output_dim, hidden_dims=[], activation=nn.ReLU, dropout=0.1):
        """
        Args:
            embed_dim (int): input embedding dim

            output_dim (int): output embedding dim

            hidden_dims (array): MLP hidden dims

            activation (nn.Module): MLP activation

            dropout (float): MLP dropout probability
        """
        super(MLP, self).__init__(embed_dim=embed_dim)

        self._output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._dropout = dropout

        self.create_layers()

    @property
    def output_dim(self):
        return self._output_dim

    def create_layers(self):
        layers = []
        prev_dim = self.embed_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation(), nn.Dropout(self._dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self._output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through MLP policy.

        Args: 
            input (tensor): data of shape [B, @self.embed_dim]

        Returns: output data (tensor) of shape [B, @self.output_dim].
        """
        return self.mlp(input)
    

class Transformer(Backbone):
    """
    Transformer policy backbone.
    """
    def __init__(self, embed_dim, **kwargs):
        """
        Args:
            embed_dim (int): input embedding dim

            kwargs (dict): args for transformer
        """
        super(Transformer, self).__init__(embed_dim=embed_dim)

        self.kwargs = kwargs

        self.create_layers()

    @classmethod
    def factory(cls, config):
        """
        Create a Transformer instance from config.

        Args:
            config (addict): config object

        Returns: Transformer instance.
        """
        return cls(
            embed_dim=config.policy.embed_dim,
            **config.policy.kwargs.backbone
        )

    @property
    def output_dim(self):
        return self.embed_dim

    def create_layers(self):
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True, norm_first=True, **self.kwargs)

    def forward(self, src, tgt):
        """
        Forward pass through transformer policy.

        Args: 
            src (tensor): the data sequence to the encoder of shape [B, T_src, @self.embed_dim]

            tgt (tensor): the data sequence to the decoder of shape [B, T_tgt, @self.embed_dim]

        Returns: output data (tensor) of shape [B, T_tgt, @self.output_dim].
        """
        return self.transformer(src, tgt)
        
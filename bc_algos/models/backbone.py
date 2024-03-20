from bc_algos.models.base_nets import PositionalEncoding
import torch.nn as nn
import math


class Backbone(nn.Module):
    """
    Abstract class for policy backbone. Subclass to implement different 
    backbone algorithms. 
    """
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): dim of input embeddings
        """
        super(Backbone, self).__init__()

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return NotImplementedError

class MLP(Backbone):
    """
    MLP policy backbone.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128,], activation=nn.ReLU, dropout=0.1):
        """
        Args:
            output_dim (int): dim of output embeddings

            hidden_dims (array-like): hidden dims of nueral net used for policy backbone

            activation (nn.Module): activation to use between linear layers

            dropout (float): dropout probability
        """
        super(MLP, self).__init__(input_dim=input_dim)

        assert len(hidden_dims) != 0, "must provide at least one hidden dim"

        self._output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._dropout = dropout

        self.create_layers()

    @property
    def output_dim(self):
        return self._output_dim

    def create_layers(self):
        self.dropout = nn.Dropout(self._dropout)
        layers = [nn.Linear(self.input_dim, self.hidden_dims[0]), self.activation()]
        for i in range(1, len(self.hidden_dims)):
            layers.extend([nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]), self.activation()])
        layers.append(nn.Linear(self.hidden_dims[-1], self._output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through MLP policy.
        """
        inputs = self.dropout(inputs)
        return self.mlp(inputs)
    

class Transformer(Backbone):
    """
    Transformer policy backbone.
    """
    def __init__(self, input_dim, nlayers, nhead, pos_embed_kwargs={}, enc_kwargs={}):
        """
        Args:
            nlayers (int): number of decoder layers

            nhead (int): number of heads in decoder layer

            pos_embed_kwargs (dict): args for positional embedding

            enc_kwargs (dict): args for transformer encoder layer
        """
        super(Transformer, self).__init__(input_dim=input_dim)

        self.nlayers = nlayers
        self.nhead = nhead
        self.pos_embed_kwargs = pos_embed_kwargs
        self.enc_kwargs = enc_kwargs

        self.create_layers()

    @property
    def output_dim(self):
        return self.input_dim

    def create_layers(self):
        self.pos_embed = PositionalEncoding(d_model=self.input_dim, **self.pos_embed_kwargs)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.nhead,
            batch_first=True,
            **self.enc_kwargs,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=self.nlayers)

    def forward(self, inputs):
        """
        Forward pass through transformer policy.
        """
        inputs = inputs * math.sqrt(self.input_dim)
        inputs = self.pos_embed(inputs)
        return self.transformer(inputs)
        
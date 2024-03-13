from bc_algos.models.base_nets import SpatialSoftArgmax
from transformers import ResNetModel
import torch.nn as nn
import torch
import numpy as np


class EncoderCore(nn.Module):
    """
    Abstract class used to encode different modailities of data.
    """
    def __init__(self, input_shape, output_shape=None, hidden_dim=[128]):
        """
        Args: 
            input_shape (array-like): shape of input excluding batch and temporal dim 

            output_shape (array-like): (optional) shape of ouput excluding batch and temporal dim 

            hidden_dim (array-like): if output_shape not None, hidden dim of nueral net used for encoding
        """
        super(EncoderCore, self).__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._project = self._output_shape is not None
        self._hidden_dim = hidden_dim

        self.create_layers()

    @property
    def output_shape(self):
        """
        Returns: output shape assuming no projection
        """
        return [np.prod(self._input_shape),]
    
    def create_layers(self):
        if self._project:
            layers = [nn.Linear(np.prod(self.output_shape), self._hidden_dim[0]), nn.ReLU()]
            for i in range(1, len(self._hidden_dim)):
                layers.extend([nn.Linear(self._hidden_dim[i-1], self._hidden_dim[i]), nn.ReLU()])
            layers.append(nn.Linear(self._hidden_dim[-1], np.prod(self._output_shape)))
            self.linear = nn.Sequential(*layers)
        else:
            self.weight = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, inputs):
        """
        Forward pass through core.
        Flatten encoded embedding and scale by learned param.
        """
        B = inputs.shape[0]
        embed = self.encode(inputs).view(B, -1)
        if self._project:
            return self.linear(embed).view(torch.Size([B]+self._output_shape))
        else:
            return self.weight * embed

    def encode(self, inputs):
        """
        Main implementation of encoder.
        For default encoder, just return input as is.
        """
        return inputs


class VisualCore(EncoderCore):
    """
    Abstract class used to encode different modailities of data.
    """
    @property
    def output_shape(self):
        return [2*self.resnet50.config.hidden_sizes[-1],]
    
    def create_layers(self):
        self.resnet50 = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.spatial_soft_argmax = SpatialSoftArgmax()

        super(VisualCore, self).create_layers()

    def encode(self, inputs):
        embed = self.resnet50(inputs).last_hidden_state
        return self.spatial_soft_argmax(embed)

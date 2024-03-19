import torchvision.transforms as transforms
from transformers import ViTMAEModel
import torch.nn as nn
import torch
import numpy as np


class EncoderCore(nn.Module):
    """
    Module used to encode low-dim data. Subclass to implement encoders for different modalities of data.
    """
    def __init__(self, input_shape, output_shape=None, hidden_dims=[128,], freeze=True, activation=nn.ReLU):
        """
        Args: 
            input_shape (array-like): shape of input excluding batch and temporal dim 

            output_shape (array-like): (optional) shape of ouput excluding batch and temporal dim 

            hidden_dims (array-like): if output_shape not None, hidden dims of nueral net used for encoding

            freeze (bool): whether to freeze backbone encoder

            activation (nn.Module): activation to use between linear layers
        """
        super(EncoderCore, self).__init__()

        assert len(hidden_dims) != 0, "must provide at least one hidden dim"

        self.input_shape = input_shape
        self._output_shape = output_shape
        self.project = output_shape is not None
        self.hidden_dims = hidden_dims
        self.freeze = freeze
        self.activation = activation

        self.create_layers()

    @property
    def enc_output_shape(self):
        """
        Returns: output shape of encoder backbone.
        """
        return self.input_shape
    
    @property
    def output_shape(self):
        """
        Returns: output shape of encoder core.
        """
        if self.project:
            return self._output_shape
        else:
            return [np.prod(self.enc_output_shape),]
    
    def create_layers(self):
        if self.project:
            layers = [nn.Linear(np.prod(self.enc_output_shape), self.hidden_dims[0]), self.activation()]
            for i in range(1, len(self.hidden_dims)):
                layers.extend([nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]), self.activation()])
            layers.append(nn.Linear(self.hidden_dims[-1], np.prod(self._output_shape)))
            self.mlp = nn.Sequential(*layers)
        else:
            self.weight = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, inputs):
        """
        Forward pass through encoder core.
        """
        B = inputs.shape[0]
        embed = self.encode(inputs).view(B, -1)
        if self.project:
            return self.mlp(embed).view(B, *self._output_shape)
        else:
            return self.weight * embed

    def encode(self, inputs):
        """
        Main implementation of encoder.
        For default encoder, just return flattened input.
        """
        return inputs


class VisualCore(EncoderCore):
    """
    EncoderCore subclass used to encode visual data.

    Adapted from "Real-World Robot Learning with Masked Visual Pre-training"
    https://arxiv.org/pdf/2210.03109.pdf. 
    """
    @property
    def enc_output_shape(self):
        return [self.vitmae.config.hidden_size,]
    
    def create_layers(self):
        self.crop = transforms.CenterCrop(224)
        self.vitmae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        # freeze resnet params
        if self.freeze:
            for param in self.vitmae.parameters():
                param.requires_grad = False

        super(VisualCore, self).create_layers()

    def encode(self, inputs):
        inputs = self.crop(inputs)
        return self.vitmae(inputs).last_hidden_state[:, 0]

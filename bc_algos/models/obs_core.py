from bc_algos.models.base_nets import SpatialSoftArgmax
from transformers import ViTMAEModel
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod


class EncoderCore(ABC, nn.Module):
    """
    Anstract class that defines necessary properties and functions to encode data. 
    Subclass to implement encoders for different modalities of data.
    """
    def __init__(self, input_shape):
        """
        Args: 
            input_shape (array-like): input shape excluding batch and temporal dim 
        """
        super(EncoderCore, self).__init__()

        self.input_shape = input_shape

    @property
    @abstractmethod
    def output_shape(self):
        """
        Returns: output shape of encoder core.
        """
        return NotImplementedError


class LowDimCore(EncoderCore):
    """
    EncoderCore subclass used to encode low-dim data.
    """
    def __init__(self, input_shape, output_shape=None, hidden_dims=[], activation=nn.ReLU):
        """
        Args: 
            input_shape (array-like): input shape excluding batch and temporal dim 

            output_shape (array-like): (optional) ouput shape

            hidden_dims (array-like): if @output_dim not None, hidden dims of mlp head

            activation (nn.Module): if @output_dim not None, activation for mlp head
        """
        super(LowDimCore, self).__init__(input_shape=input_shape)

        self._output_shape = output_shape
        self.project = output_shape is not None
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.create_layers()
    
    @property
    def output_shape(self):
        """
        Returns: output shape of low-dim encoder core.
        """
        if self.project:
            return self._output_shape
        else:
            return [np.prod(self.input_shape),]
        
    def create_layers(self):
        if self.project:
            layers = []
            prev_dim = np.prod(self.input_shape)
            for hidden_dim in self.hidden_dims:
                layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation()])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, np.prod(self._output_shape)))
            self.mlp = nn.Sequential(*layers)
        else:
            self.weight = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, inputs):
        """
        Forward pass through low-dim encoder core.
        """
        B = inputs.shape[0]
        inputs = inputs.view(B, -1)
        if self.project:
            return self.mlp(inputs).view(B, *self._output_shape)
        else:
            return self.weight * inputs


class ViTMAECore(EncoderCore):
    """
    EncoderCore subclass used to encode visual data with VitMAE backbone.

    Adapted from "Real-World Robot Learning with Masked Visual Pre-training"
    https://arxiv.org/pdf/2210.03109.pdf. 
    """
    def __init__(self, input_shape, freeze=True):
        """
        Args: 
            input_shape (array-like): input shape excluding batch and temporal dim 

            freeze (bool): whether or not to freeze VitMAE backbone
        """
        super(ViTMAECore, self).__init__(input_shape=input_shape)

        self.create_layers()

        if freeze:
            self.freeze()

    @property
    def output_shape(self):
        """
        Returns: output shape of ViTMAE encoder core.
        """
        return [self.vitmae.config.hidden_size,]
    
    def freeze(self):
        """
        Freeze encoder network parameters.
        """
        for param in self.vitmae.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        self.vitmae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

    def forward(self, inputs):
        """
        Forward pass through ViTMAE encoder core.
        """
        return self.vitmae(inputs).last_hidden_state[:, 0]
    

class ResNet18Core(EncoderCore):
    """
    EncoderCore subclass used to encode visual data with ResNet-18 backbone.
    """
    def __init__(self, input_shape, freeze=True):
        """
        Args: 
            input_shape (array-like): input shape excluding batch and temporal dim 

            freeze (bool): whether or not to freeze ResNet-18 backbone
        """
        super(ResNet18Core, self).__init__(input_shape=input_shape)

        self.create_layers()

        if freeze:
            self.freeze()

    @property
    def output_shape(self):
        """
        Returns: output shape of ResNet-18 encoder core.
        """
        return [512, 2]
    
    def freeze(self):
        """
        Freeze encoder network parameters.
        """
        for param in self.resnet18.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        resnet18_classifier = models.resnet18(pretrained=True)
        # remove pooling and fc layers
        self.resnet18 = torch.nn.Sequential(*(list(resnet18_classifier.children())[:-2]))
        self.spatial_softmax = SpatialSoftArgmax()

    def forward(self, inputs):
        """
        Forward pass through ResNet-18 encoder core.
        """
        B = inputs.shape[0]
        return self.spatial_softmax(self.resnet18(inputs)).view(B, *self.output_shape)

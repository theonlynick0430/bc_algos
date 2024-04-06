from bc_algos.models.base_nets import pos_enc_2d
import bc_algos.utils.constants as Const
from transformers import ViTMAEModel
import torchvision.models as models
from torchvision.transforms import Normalize
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
            input_shape (array-like): input shape excluding batch dim 
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
            input_shape (array-like): input shape excluding batch dim 

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
            return self.mlp(inputs).view(-1, *self._output_shape)
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
            input_shape (array-like): input shape excluding batch dim 

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
        for param in self.network.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        preprocessor = Normalize(mean=Const.IMAGE_NET_MEAN, std=Const.IMAGE_NET_STD)
        self.vitmae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.network = nn.Sequential(preprocessor, self.vitmae)

    def forward(self, inputs):
        """
        Forward pass through ViTMAE encoder core.
        """
        return self.network(inputs).last_hidden_state[:, 0]
    

class ResNet18Core(EncoderCore):
    """
    EncoderCore subclass used to encode visual data with ResNet-18 backbone.
    """
    def __init__(self, input_shape, embed_shape=[512, 8, 8,], freeze=True):
        """
        Args: 
            input_shape (array-like): input shape excluding batch dim 

            embed_shape (array-like): output shape of ResNet-18 backbone excluding batch dim. 
                Defaults to output shape for inputs images of resolution 256x256.

            freeze (bool): whether or not to freeze ResNet-18 backbone
        """
        super(ResNet18Core, self).__init__(input_shape=input_shape)
        self.embed_shape = embed_shape

        self.create_layers()

        if freeze:
            self.freeze()

    @property
    def output_shape(self):
        """
        Returns: output shape of ResNet-18 encoder core.
        """
        C, H, W = self.embed_shape
        return [H*W, C,]
    
    def freeze(self):
        """
        Freeze encoder network parameters.
        """
        for param in self.network.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        preprocessor = Normalize(mean=Const.IMAGE_NET_MEAN, std=Const.IMAGE_NET_STD)
        resnet18_classifier = models.resnet18(pretrained=True)
        # remove pooling and fc layers
        resnet18 = torch.nn.Sequential(*(list(resnet18_classifier.children())[:-2]))
        self.network = nn.Sequential(preprocessor, resnet18)

    def forward(self, inputs):
        """
        Forward pass through ResNet-18 encoder core.
        """
        device = inputs.device
        C, H, W = self.embed_shape
        embed = self.network(inputs)
        embed += pos_enc_2d(d_model=C, H=H, W=W, device=device)
        return torch.transpose(embed.view(-1, C, H*W), -1, -2).contiguous()

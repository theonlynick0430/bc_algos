from bc_algos.models.base_nets import pos_enc_2d, SpatialSoftArgmax
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
            input_shape (array): input shape excluding batch dim 
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
            input_shape (array): input shape excluding batch dim 

            output_shape (array): (optional) ouput shape

            hidden_dims (array): if @output_dim not None, hidden dims of mlp head

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
            return [np.prod(self.input_shape)]
        
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

    def forward(self, input):
        """
        Forward pass through low-dim encoder core.

        Args:
            input (tensor): data of shape [B, @self.input_shape]

        Returns: output data (tensor) of shape [B, @self.output_shape].
        """
        B = input.shape[0]
        input = input.view(B, -1)
        if self.project:
            return self.mlp(input).view(-1, *self._output_shape)
        else:
            return self.weight * input


class ViTMAECore(EncoderCore):
    """
    EncoderCore subclass used to encode visual data with VitMAE backbone.

    Adapted from "Real-World Robot Learning with Masked Visual Pre-training"
    https://arxiv.org/pdf/2210.03109.pdf. 
    """
    def __init__(self, input_shape, freeze=True):
        """
        Args: 
            input_shape (array): input shape excluding batch dim. 
                Expected to follow [B, C, H, W].

            freeze (bool): if True, freeze VitMAE backbone
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
        return [self.vitmae.config.hidden_size]
    
    def freeze(self):
        """
        Freeze encoder network parameters.
        """
        for param in self.vitmae.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        self.preprocessor = Normalize(mean=Const.IMAGE_NET_MEAN, std=Const.IMAGE_NET_STD)
        self.vitmae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

    def forward(self, input):
        """
        Forward pass through ViTMAE encoder core.

        Args:
            input (tensor): data of shape [B, @self.input_shape]

        Returns: output data (tensor) of shape [B, @self.output_shape].
        """
        input = self.preprocessor(input)
        return self.vitmae(input).last_hidden_state[:, 0]
    

class ResNet18Core(EncoderCore):
    """
    EncoderCore subclass used to encode visual data with ResNet-18 backbone.
    """
    def __init__(self, input_shape, embed_shape=[512, 8, 8], spatial_softmax=False, freeze=False):
        """
        Args: 
            input_shape (array): input shape excluding batch dim.
                Expected to follow [B, C, H, W].

            embed_shape (array): output shape of ResNet-18 backbone excluding batch dim. 
                Defaults to output shape for input images of resolution 256x256.

            spatial_softmax (bool): if True, use SpatialSoftArgmax 

            freeze (bool): if True, freeze ResNet-18 backbone
        """
        super(ResNet18Core, self).__init__(input_shape=input_shape)
        self.embed_shape = embed_shape
        self.spat_soft = spatial_softmax

        self.create_layers()

        if freeze:
            self.freeze()

    @property
    def output_shape(self):
        """
        Returns: output shape of ResNet-18 encoder core.
        """
        C, H, W = self.embed_shape
        return [2, C]
    
    def freeze(self):
        """
        Freeze encoder network parameters.
        """
        for param in self.resnet18.parameters():
            param.requires_grad = False
    
    def create_layers(self):
        self.preprocessor = Normalize(mean=Const.IMAGE_NET_MEAN, std=Const.IMAGE_NET_STD)
        resnet18_classifier = models.resnet18(pretrained=True)
        # remove pooling and fc layers
        self.resnet18 = nn.Sequential(*list(resnet18_classifier.children())[:-2])
        if self.spat_soft:
            self.spatial_softmax = SpatialSoftArgmax(normalize=True)
        else:
            C, H, W = self.embed_shape
            self.pos_enc = nn.Parameter(pos_enc_2d(d_model=C, H=H, W=W))
            self.pos_enc.requires_grad = False # buffer

    def forward(self, input):
        """
        Forward pass through ResNet-18 encoder core.

        Args:
            input (tensor): data of shape [B, @self.input_shape]

        Returns: output data (tensor) of shape [B, @self.output_shape].
        """
        C, H, W = self.embed_shape
        input = self.preprocessor(input)
        latent = self.resnet18(input)
        if self.spat_soft:
            latent = self.spatial_softmax(latent)
            latent = latent.view(-1, C, 2)
        else:
            latent = latent + self.pos_enc
            latent = latent.view(-1, C, H*W)
        return torch.transpose(latent, -1, -2).contiguous()

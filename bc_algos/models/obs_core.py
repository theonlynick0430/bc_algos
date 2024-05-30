from bc_algos.models.base_nets import pos_enc_2d, SpatialSoftArgmax
from bc_algos.models.token_learner import TokenLearnerConv, TokenLearnerMLP
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
    def __init__(self, input_shape, output_shape, hidden_dims=[], activation=nn.ReLU, dropout=0.1):
        """
        Args: 
            input_shape (array): input shape excluding batch dim 

            output_shape (array): output shape excluding batch dim 

            hidden_dims (array): MLP hidden dims

            activation (nn.Module): MLP activation

            dropout (float): MLP dropout probability
        """
        super(LowDimCore, self).__init__(input_shape=input_shape)

        self._output_shape = output_shape
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._dropout = dropout

        self.create_layers()
    
    @property
    def output_shape(self):
        """
        Returns: output shape of low-dim encoder core.
        """
        return self._output_shape
        
    def create_layers(self):
        layers = []
        prev_dim = np.prod(self.input_shape)
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation(), nn.Dropout(self._dropout)])
            prev_dim = hidden_dim
        layers.extend([nn.Linear(prev_dim, np.prod(self._output_shape))])
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through low-dim encoder core.

        Args:
            input (tensor): data of shape [B, @self.input_shape]

        Returns: output data (tensor) of shape [B, @self.output_shape].
        """
        B = input.shape[0]
        input = input.view(B, -1)
        return self.mlp(input).view(-1, *self._output_shape)


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
                Expected to follow [C, H, W].

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
    def __init__(
        self, 
        input_shape, 
        embed_shape=[512, 8, 12], 
        freeze=True,
        spatial_reduction=None, 
        spatial_reduction_args=None,
    ):
        """
        Args: 
            input_shape (array): input shape excluding batch dim.
                Expected to follow [C, H, W].

            embed_shape (array): output shape of ResNet-18 backbone excluding batch dim. 
                Defaults to output shape for input images of resolution 256x384.

            freeze (bool): if True, freeze ResNet-18 backbone

            spatial_reduction (SpatialReduction): (optional) type of spatial reduction 

            spatial_reduction_args (dict): arguments for spatial reduction
        """
        super(ResNet18Core, self).__init__(input_shape=input_shape)
        self.embed_shape = embed_shape
        self.spatial_reduction = spatial_reduction
        self.spatial_reduction_args = spatial_reduction_args

        self.create_layers()

        if freeze:
            self.freeze()

    @property
    def output_shape(self):
        """
        Returns: output shape of ResNet-18 encoder core.
        """
        C, H, W = self.embed_shape
        if self.spatial_reduction == Const.SpatialReduction.SPATIAL_SOFTMAX:
            return [2, C]
        elif (self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_MLP or
              self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_CONV):
            return [self.token_learner.S, C]
        else:
            return [H*W, C]
    
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

        if self.spatial_reduction == Const.SpatialReduction.SPATIAL_SOFTMAX:
            self.spatial_softmax = SpatialSoftArgmax(normalize=True)
        elif self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_MLP:
            self.token_learner = TokenLearnerMLP(input_shape=self.embed_shape, **self.spatial_reduction_args)
        elif self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_CONV:
            self.token_learner = TokenLearnerConv(input_shape=self.embed_shape, **self.spatial_reduction_args)
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

        if self.spatial_reduction == Const.SpatialReduction.SPATIAL_SOFTMAX:
            return self.spatial_softmax(latent)
        elif (self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_MLP or
              self.spatial_reduction == Const.SpatialReduction.TOKEN_LEARNER_CONV):
            return self.token_learner(latent)
        else:
            latent = latent + self.pos_enc
            latent = latent.view(-1, C, H*W)
            return torch.transpose(latent, -1, -2).contiguous()

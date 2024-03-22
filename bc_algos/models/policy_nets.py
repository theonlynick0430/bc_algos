from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Backbone, MLP, Transformer
import bc_algos.utils.tensor_utils as TensorUtils
import torch.nn as nn


class BC(nn.Module):
    """
    Abstract class for behavorial cloning policy that predicts actions from observations.
    Subclass to implement different behavorial cloning policies.
    """
    def __init__(self, obs_group_enc, backbone, act_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            act_dec (ActionDecoder): output decoder
        """
        super(BC, self).__init__()

        assert isinstance(obs_group_enc, ObservationGroupEncoder)
        assert isinstance(backbone, Backbone)
        assert isinstance(act_dec, ActionDecoder)

        self.obs_group_enc = obs_group_enc
        self.backbone = backbone
        self.act_dec = act_dec


class BC_MLP(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations using MLP.
    """
    def __init__(self, obs_group_enc, backbone, act_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            act_dec (ActionDecoder): output decoder
        """
        super(BC_MLP, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, act_dec=act_dec)

        assert isinstance(backbone, MLP)

        self.nets = nn.Sequential(obs_group_enc, backbone, act_dec)

    def forward(self, inputs):
        """
        Forward pass through BC_MLP.

        Args: 
            inputs (dict): nested dictionary that maps observation group to observation key
            to data of shape [B, T=1, D,]

        Returns: action in shape [B, T=1, action_dim,]
        """
        return TensorUtils.time_distributed(inputs=inputs, op=self.nets)
    

class BC_Transformer(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations using transformer.
    """
    def __init__(self, obs_group_enc, backbone, act_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            act_dec (ActionDecoder): output decoder
        """
        super(BC_Transformer, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, act_dec=act_dec)

        assert isinstance(backbone, Transformer)

    def forward(self, inputs):
        """
        Forward pass through BC_Transformer.

        Args: 
            inputs (dict): nested dictionary that maps observation group to observation key
            to data of shape [B, T, D,]

        Returns: actions in shape [B, T, action_dim,]
        """
        embed = TensorUtils.time_distributed(inputs=inputs, op=self.obs_group_enc)
        embed = self.backbone(embed)
        actions = TensorUtils.time_distributed(inputs=embed, op=self.act_dec)
        return actions

from bc_algos.models.base_nets import pos_enc_1d
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

    @classmethod
    def prepare_inputs(cls, inputs, device=None):
        """
        Prepares inputs to be processed by model by converting to float tensor, and moving 
        to specified device.

        Args:
            inputs (dict): nested dictionary that maps observation group to observation key
                to data of shape [B, T, ...]

            device: (optional) device to send tensors to

        Returns: prepared inputs.
        """
        inputs = TensorUtils.to_tensor(x=inputs, device=device)
        return TensorUtils.to_float(x=inputs)


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
                to data of shape [B, T=1, ...]

        Returns: action (tensor) in shape [B, T=1, action_dim,].
        """
        return TensorUtils.time_distributed(inputs=inputs, op=self.nets)
    

class BC_Transformer(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations using transformer.
    """
    def __init__(self, obs_group_enc, backbone, act_dec, act_chunk):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            act_dec (ActionDecoder): output decoder

            act_chunk (int): number of actions to predict in a single model pass
        """
        super(BC_Transformer, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, act_dec=act_dec)

        assert isinstance(backbone, Transformer)

        self.act_chunk = act_chunk
        self.embed_dim = backbone.embed_dim

    def forward(self, inputs):
        """
        Forward pass through BC_Transformer.

        Args: 
            inputs (dict): nested dictionary that maps observation group to observation key
                to data of shape [B, T, ...]

        Returns: actions (tensor) in shape [B, T, action_dim,].
        """
        src = TensorUtils.time_distributed(inputs=inputs, op=self.obs_group_enc)
        device = src.device
        B, T, _ = src.shape
        src = src.view(B, T, -1, self.embed_dim)
        _, _, N, _ = src.shape
        src += pos_enc_1d(d_model=self.embed_dim, T=T, device=device).unsqueeze(1).repeat(1, N, 1)
        src = src.view(B, -1, self.embed_dim)
        tgt = pos_enc_1d(d_model=self.embed_dim, T=self.act_chunk, device=device).unsqueeze(0).repeat(B, 1, 1)
        embed = self.backbone(src, tgt)
        actions = TensorUtils.time_distributed(inputs=embed, op=self.act_dec)
        return actions

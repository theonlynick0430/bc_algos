from bc_algos.models.base_nets import pos_enc_1d
from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Backbone, MLP, Transformer
import bc_algos.utils.tensor_utils as TensorUtils
import torch.nn as nn
import torch


class BC(nn.Module):
    """
    Abstract class for behavorial cloning policy that predicts actions from observations.
    Subclass to implement different behavorial cloning policies.
    """
    def __init__(self, obs_group_enc, backbone, action_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder
        """
        super(BC, self).__init__()

        assert isinstance(obs_group_enc, ObservationGroupEncoder)
        assert isinstance(backbone, Backbone)
        assert isinstance(action_dec, ActionDecoder)

        self.obs_group_enc = obs_group_enc
        self.backbone = backbone
        self.action_dec = action_dec

    @classmethod
    def prepare_inputs(cls, inputs, device=None):
        """
        Prepare inputs to be processed by model by converting to float tensor
        and moving to specified device.

        Args:
            inputs (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, ...]

            device: (optional) device to send tensors to

        Returns: prepared inputs.
        """
        inputs = TensorUtils.to_tensor(x=inputs, device=device)
        return TensorUtils.to_float(x=inputs)


class BC_MLP(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations using MLP.
    """
    def __init__(self, obs_group_enc, backbone, action_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder
        """
        super(BC_MLP, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, action_dec=action_dec)

        assert isinstance(backbone, MLP)

    def forward(self, inputs):
        """
        Forward pass through BC_MLP.

        Args: 
            inputs (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, T=1, ...]

        Returns: actions (tensor) of shape [B, T=1, action_dim].
        """
        latent_dict = TensorUtils.time_distributed(inputs=inputs, op=self.obs_group_enc)
        latents = torch.cat(list(latent_dict.values()), dim=-1)
        outputs = TensorUtils.time_distributed(inputs=latents, op=self.backbone)
        actions = TensorUtils.time_distributed(inputs=outputs, op=self.action_dec)
        return actions
    

class BC_Transformer(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations 
    using transformer encoder-decoder architecture.
    """
    def __init__(self, obs_group_enc, backbone, action_dec, action_chunk):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder

            action_chunk (int): number of actions to predict in a single model pass
        """
        super(BC_Transformer, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, action_dec=action_dec)

        assert isinstance(backbone, Transformer)

        self.action_chunk = action_chunk
        self.embed_dim = backbone.embed_dim

    @classmethod
    def factory(cls, config, obs_group_enc, backbone, action_dec):
        """
        Create a BC_Transformer instance from config.

        Args:
            config (addict): config object

            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder

        Returns: BC_Transformer instance.
        """
        return cls(
            obs_group_enc=obs_group_enc,
            backbone=backbone,
            action_dec=action_dec,
            action_chunk=config.dataset.seq_length,
        )

    def prepare_latent(self, latent):
        """
        Prepare latent for transformer by reshaping from [B, T, N * @self.embed_dim] 
        to [B, T*N, @self.embed_dim] and adding positional embeddings.

        Args: 
            latent (tensor): data of shape [B, T, N * @self.embed_dim]

        Returns: prepared latent (tensor) of shape [B, T*N, @self.embed_dim].
        """
        device = latent.device
        B, T, _ = latent.shape
        if T > 1:
            latent = latent.view(B, T, -1, self.embed_dim)
            _, _, N, _ = latent.shape
            latent += pos_enc_1d(d_model=self.embed_dim, T=T, device=device).unsqueeze(1).repeat(1, N, 1)
        return latent.view(B, -1, self.embed_dim)

    def forward(self, inputs):
        """
        Forward pass through BC_Transformer.

        Args: 
            inputs (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, T_obs/T_goal, ...]

        Returns: actions (tensor) of shape [B, T=@self.action_chunk, action_dim].
        """
        latent_dict = TensorUtils.time_distributed(inputs=inputs, op=self.obs_group_enc)
        src = torch.cat([self.prepare_latent(latent) for latent in latent_dict.values()], dim=-2)
        device = src.device
        B = src.shape[0]
        tgt = pos_enc_1d(d_model=self.embed_dim, T=self.action_chunk, device=device).unsqueeze(0).repeat(B, 1, 1)
        outputs = self.backbone(src, tgt)
        actions = TensorUtils.time_distributed(inputs=outputs, op=self.action_dec)
        return actions

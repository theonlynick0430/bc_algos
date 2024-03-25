import bc_algos.utils.obs_utils as ObsUtils
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np 


class ObservationEncoder(nn.Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with encoder networks.
    """
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        # maps observation key to encoder core
        # ex: {"robot0_eef_pos": EncoderCore, "agentview_image": VisualCore}
        self.obs_key_to_enc_core = nn.ModuleDict()

    def register_obs_key(self, obs_key, modality, input_shape, **kwargs):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            obs_key (str): observation key to register

            modality (Modality): modality of @obs_key

            input_shape (array-like): shape of data corresponding to @obs_key excluding batch and temporal dim

            kwargs (dict): arguments for encoder core
        """
        assert obs_key not in ObsUtils.MODALITY_TO_ENC_CORE, f"observation key {obs_key} already registered"
        assert modality in ObsUtils.MODALITY_TO_ENC_CORE, f"modality {modality} not found in MODALITY_TO_ENC_CORE"
        enc_core = ObsUtils.MODALITY_TO_ENC_CORE[modality](input_shape=input_shape, **kwargs)
        self.obs_key_to_enc_core[obs_key] = enc_core

    @property
    def output_dim(self):
        """
        Returns: output dim of observation encoder.
        """
        dim = 0
        for enc_core in self.obs_key_to_enc_core.values():
            dim += np.prod(enc_core.output_shape)
        return dim
    
    def forward(self, inputs):
        """
        Forward pass through observation encoder.
        """
        feats = []
        for obs_key in inputs:
            if obs_key in self.obs_key_to_enc_core:
                embed = self.obs_key_to_enc_core[obs_key](inputs[obs_key])
                B = embed.shape[0]
                feats.append(embed.view(B, -1))
        return torch.cat(feats, dim=-1)


class ObservationGroupEncoder(nn.Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.
    """
    def __init__(self):
        super(ObservationGroupEncoder, self).__init__()

        # maps observation group to observation core
        # ex: {"obs": ObservationEncoder, "goal": ObservationEncoder}
        self.obs_group_to_obs_enc = nn.ModuleDict()

    @classmethod
    def factory(cls, config):
        """
        Create a ObservationGroupEncoder instance from config.

        Args:
            config (addict): config object

        Returns:
            ObservationGroupEncoder instance
        """
        obs_group_enc = cls()
        for obs_group, obs_keys in ObsUtils.OBS_GROUP_TO_KEY.items():
            obs_enc = ObservationEncoder()
            for obs_key in obs_keys:
                shape = ObsUtils.OBS_KEY_TO_SHAPE[obs_key]
                modality = ObsUtils.OBS_KEY_TO_MODALITY[obs_key]
                obs_enc.register_obs_key(
                    obs_key=obs_key,
                    modality=modality,
                    input_shape=shape,
                    **config.observation.kwargs[modality]
                )
            obs_group_enc.register_obs_group(obs_group=obs_group, obs_enc=obs_enc)
        return obs_group_enc

    def register_obs_group(self, obs_group, obs_enc):
        assert obs_group not in self.obs_group_to_obs_enc, f"observation group {obs_group} already registered"
        assert isinstance(obs_enc, ObservationEncoder)
        self.obs_group_to_obs_enc[obs_group] = obs_enc

    @property
    def output_dim(self):
        """
        Returns: output dim of observation group encoder.
        """
        dim = 0
        for obs_enc in self.obs_group_to_obs_enc.values():
            dim += obs_enc.output_dim
        return dim

    def forward(self, inputs):
        """
        Forward pass through observation group encoder.
        """
        feats = []
        for obs_group in inputs:
            if obs_group in self.obs_group_to_obs_enc:
                embed = self.obs_group_to_obs_enc[obs_group](inputs[obs_group])
                feats.append(embed)
        return torch.cat(feats, dim=-1)
    

class ActionDecoder(nn.Module):
    """
    Module that can generate action outputs using nueral network. Inputs are assumed
    to be flat (usually outputs from policy backbone). Subclass this module for
    implementing more complex schemes.
    """
    def __init__(
        self,
        action_shape,
        input_dim,
        hidden_dims=[128,],
        activation=nn.ReLU
    ):
        """
        Args:
            action_shape (int): shape of single action

            input_dim (int): input dim

            hidden_dims (array-like): hidden dims of nueral net used for decoding

            activation (nn.Module): activation to use between linear layers
        """
        super(ActionDecoder, self).__init__()

        assert len(hidden_dims) != 0, "must provide at least one hidden dim"

        self.action_shape = action_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.create_layers()

    @classmethod
    def factory(cls, config, input_dim):
        """
        Create a ActionDecoder instance from config.

        Args:
            config (addict): config object

            input_dim (int): dim of input embeddings

        Returns:
            ActionDecoder instance
        """
        return cls(
            action_shape=config.policy.action_shape,
            input_dim=input_dim,
            **config.policy.kwargs.action_decoder
        )

    def create_layers(self):
        layers = [nn.Linear(self.input_dim, self.hidden_dims[0]), self.activation()]
        for i in range(1, len(self.hidden_dims)):
            layers.extend([nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]), self.activation()])
        layers.append(nn.Linear(self.hidden_dims[-1], np.prod(self.action_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through action decoder.
        """
        B = inputs.shape[0]
        return self.mlp(inputs).view(B, *self.action_shape)

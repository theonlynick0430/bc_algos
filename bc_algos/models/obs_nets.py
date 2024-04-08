import bc_algos.utils.obs_utils as ObsUtils
import torch
import torch.nn as nn
import numpy as np 


class ObservationEncoder(nn.Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    """
    def __init__(self, obs_keys):
        """
        Args:
            obs_keys (list): observation keys to encode
        """
        super(ObservationEncoder, self).__init__()

        # maps observation key to encoder core
        # ex: {"robot0_eef_pos": LowDimCore(), "robot0_eef_quat": LowDimCore(), "agentview_image": ResNet18Core()}
        self.obs_key_to_enc_core = nn.ModuleDict()

        for obs_key in obs_keys:
            self.obs_key_to_enc_core[obs_key] = ObsUtils.OBS_KEY_TO_ENC_CORE[obs_key]

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

        Args:
            inputs (dict): nested dictionary that maps observation key
                to data of shape [B, ...]

        Returns: output data (tensor) with shape [B, @self.output_dim].
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
    def __init__(self, obs_group_to_key):
        """
        Args:
            obs_group_to_key (dict): dictionary from observation group to observation key
        """
        super(ObservationGroupEncoder, self).__init__()

        # maps observation group to observation core
        # ex: {"obs": ObservationEncoder(), "goal": ObservationEncoder()}
        self.obs_group_to_obs_enc = nn.ModuleDict()

        for obs_group in obs_group_to_key:
            self.obs_group_to_obs_enc[obs_group] = ObservationEncoder(obs_keys=obs_group_to_key[obs_group])

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

        Args:
            inputs (dict): nested dictionary that maps observation group to observation key
                to data of shape [B, ...]

        Returns: output data (tensor) with shape [B, @self.output_dim].
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
        hidden_dims=[],
        activation=nn.ReLU
    ):
        """
        Args:
            action_shape (int): shape of single action

            input_dim (int): input dim

            hidden_dims (array): hidden dims of nueral net used for decoding

            activation (nn.Module): activation to use between linear layers
        """
        super(ActionDecoder, self).__init__()

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
        layers = []
        prev_dim = np.prod(self.input_dim)
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, np.prod(self.action_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through action decoder.

        Args:
            inputs (tensor): data with shape [B, @self.input_shape]

        Returns: output data (tensor) with shape [B, @self.output_shape].
        """
        return self.mlp(inputs).view(-1, *self.action_shape)

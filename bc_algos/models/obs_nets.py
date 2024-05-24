import bc_algos.utils.obs_utils as ObsUtils
import torch
import torch.nn as nn
import numpy as np 
from collections import OrderedDict


class ObservationEncoder(nn.Module):
    """
    Module that processes input by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    """
    def __init__(self, obs_keys):
        """
        Args:
            obs_keys (array): observation keys to encode
        """
        super(ObservationEncoder, self).__init__()

        # module dictionary from observation key to encoder core.
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
    
    def forward(self, input):
        """
        Forward pass through observation encoder.

        Args:
            input (dict): nested dictionary that maps observation key
                to data (tensor) of shape [B, ...]

        Returns: output data (tensor) of shape [B, @self.output_dim].
        """
        latents = []
        for obs_key in input:
            if obs_key in self.obs_key_to_enc_core:
                latent = self.obs_key_to_enc_core[obs_key](input[obs_key])
                B = latent.shape[0]
                latents.append(latent.view(B, -1))
        return torch.cat(latents, dim=-1)


class ObservationGroupEncoder(nn.Module):
    """
    This class allows networks to encode multiple observation dictionaries by 
    assigning each observation group an @ObservationEncoder object.
    """
    def __init__(self, obs_group_to_key):
        """
        Args:
            obs_group_to_key (dict): dictionary from observation group to observation key
        """
        super(ObservationGroupEncoder, self).__init__()

        # module dictionary from observation group to observation core
        self.obs_group_to_obs_enc = nn.ModuleDict()

        for obs_group in obs_group_to_key:
            self.obs_group_to_obs_enc[obs_group] = ObservationEncoder(obs_keys=obs_group_to_key[obs_group])

    @classmethod
    def factory(cls, config):
        """
        Create a ObservationGroupEncoder instance from config.

        Args:
            config (addict): config object

        Returns: ObservationGroupEncoder instance.
        """
        return cls(obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY)

    @property
    def output_dim(self):
        """
        Returns: dictionary from observation group to output dim of 
            corresponding observation encoder
        """
        return {k: self.obs_group_to_obs_enc[k].output_dim for k in self.obs_group_to_obs_enc}

    def forward(self, input):
        """
        Forward pass through observation group encoder.

        Args:
            input (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, ...]

        Returns: dictionary from observation group: obs_group to data (tensor) 
            of shape [B, @self.output_dim[obs_group]].
        """
        latent_dict = OrderedDict()
        for obs_group in input:
            if obs_group in self.obs_group_to_obs_enc:
                latent_dict[obs_group] = self.obs_group_to_obs_enc[obs_group](input[obs_group])
        return latent_dict
    

class ActionDecoder(nn.Module):
    """
    Module that can generate action output using nueral network. input are assumed
    to be flat (usually output from policy backbone). Subclass this module for
    implementing more complex schemes.
    """
    def __init__(
        self,
        action_shape,
        input_dim,
        hidden_dims=[],
        activation=nn.ReLU,
        dropout=0.1,
    ):
        """
        Args:
            action_shape (int): shape of single action

            input_dim (int): input embedding dim

            hidden_dims (array): MLP hidden dims

            activation (nn.Module): MLP activation

            dropout (float): MLP dropout probability
        """
        super(ActionDecoder, self).__init__()

        self.action_shape = action_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._dropout = dropout

        self.create_layers()

    @classmethod
    def factory(cls, config, input_dim):
        """
        Create a ActionDecoder instance from config.

        Args:
            config (addict): config object

            input_dim (int): input embedding dim

        Returns: ActionDecoder instance.
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
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation(), nn.Dropout(self._dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, np.prod(self.action_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through action decoder.

        Args:
            input (tensor): data of shape [B, @self.input_shape]

        Returns: output data (tensor) of shape [B, @self.output_shape].
        """
        return self.mlp(input).view(-1, *self.action_shape)

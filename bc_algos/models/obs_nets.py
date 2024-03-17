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
        super().__init__()

        # maps observation key to encoder core
        # ex: {"robot0_eef_pos": EncoderCore, "agentview_image": VisualCore}
        self.obs_key_to_enc_core = OrderedDict()

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
    def output_shape(self):
        """
        Returns: output shape of observation encoder.
        """
        dim = 0
        for enc_core in self.obs_key_to_enc_core.values():
            dim += np.prod(enc_core.output_shape)
        return [dim,]
    
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
        super().__init__()

        # maps observation group to observation core
        # ex: {"obs": ObservationEncoder, "goal": ObservationEncoder}
        self.obs_group_to_obs_enc = OrderedDict()

    def register_obs_group(self, obs_group, obs_enc):
        assert obs_group not in self.obs_group_to_obs_enc, f"observation group {obs_group} already registered"
        assert isinstance(obs_enc, ObservationEncoder)
        self.obs_group_to_obs_enc[obs_group] = obs_enc

    @property
    def output_shape(self):
        """
        Returns: output shape of observation group encoder.
        """
        dim = 0
        for obs_enc in self.obs_group_to_obs_enc.values():
            dim += obs_enc.output_shape[0]
        return [dim,]

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


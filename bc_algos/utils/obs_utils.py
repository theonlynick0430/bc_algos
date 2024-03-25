from collections import OrderedDict
from enum import Enum


class Modality(str, Enum):
    LOW_DIM="low_dim"
    RGB="rgb"
    DEPTH="depth"

# maps modality to encoder core class
# Ex: {Modality.LOW_DIM: EncoderCore, Modality.RGB: VisualCore}
MODALITY_TO_ENC_CORE = OrderedDict()

# maps observation key to shape
# Ex: {"robot0_eef_pos": [3,], "agentview_image": [3, 224, 224,]}
OBS_KEY_TO_SHAPE = OrderedDict()

# maps observation key to modality
# Ex: {"robot0_eef_pos": Modality.LOW_DIM, "agentview_image": Modality.RGB}
OBS_KEY_TO_MODALITY = OrderedDict()

# maps observation group to observation key
# Ex: {"obs": ["robot0_eef_pos", "robot0_eef_quat",], "goal": ["agentview_image",]}
OBS_GROUP_TO_KEY = OrderedDict()

def register_encoder_core(core, modality):
    assert modality not in MODALITY_TO_ENC_CORE, f"modality {modality} already registered"
    MODALITY_TO_ENC_CORE[modality] = core

def unregister_encoder_core(modality):
    assert modality in MODALITY_TO_ENC_CORE, f"modality {modality} not registered"
    MODALITY_TO_ENC_CORE.pop(modality)

def init_obs_utils(config):
    """
    Initialize mappings from config.

    Args:
        config (addict): config object
    """
    for obs_key, shape in config.observation.shapes.items():
        OBS_KEY_TO_SHAPE[obs_key] = shape
    for obs_group, modality_to_obs_key in config.observation.modalities.items():
        for modality, obs_keys in modality_to_obs_key.items():
            for obs_key in obs_keys:
                if obs_group in OBS_GROUP_TO_KEY:
                    OBS_GROUP_TO_KEY[obs_group].append(obs_key)
                else:
                    OBS_GROUP_TO_KEY[obs_group] = [obs_key,]
                if obs_key not in OBS_KEY_TO_MODALITY:
                    OBS_KEY_TO_MODALITY[obs_key] = modality
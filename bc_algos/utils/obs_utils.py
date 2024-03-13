from collections import OrderedDict
from enum import Enum


class Modality(Enum):
    LOW_DIM="low_dim"
    RGB="rgb"

# Maps modality to encoder core class
# Ex: {Modality.LOW_DIM: EncoderCore, Modality.RGB: VisualCore}
MODALITY_TO_ENC_CORE = OrderedDict()

# Maps observation key to modality
# Ex: {"robot0_eef_pos": Modality.LOW_DIM, "agentview_image": Modality.RGB}
OBS_KEY_TO_MODALITY = OrderedDict()

# Maps observation group to observation key to shape
# Ex: {"obs": {"robot0_eef_pos": [3]}, "goal": {"agentview_image": [3, 84, 84]}}
INPUT_SHAPES = OrderedDict()

def register_obs_key(obs_key, modality):
    assert obs_key not in OBS_KEY_TO_MODALITY, f"Already registered observation key {obs_key}!"
    OBS_KEY_TO_MODALITY[obs_key] = modality

def register_encoder_core(core, modality):
    assert modality not in MODALITY_TO_ENC_CORE, f"Already registered modality {modality}!"
    MODALITY_TO_ENC_CORE[modality] = core
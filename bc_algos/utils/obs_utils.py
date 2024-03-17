from collections import OrderedDict
from enum import Enum


class Modality(Enum):
    LOW_DIM="low_dim"
    RGB="rgb"

# maps modality to encoder core class
# Ex: {Modality.LOW_DIM: EncoderCore, Modality.RGB: VisualCore}
MODALITY_TO_ENC_CORE = OrderedDict()

def register_encoder_core(core, modality):
    assert modality not in MODALITY_TO_ENC_CORE, f"modality {modality} already registered"
    MODALITY_TO_ENC_CORE[modality] = core

def unregister_encoder_core(modality):
    assert modality in MODALITY_TO_ENC_CORE, f"modality {modality} not registered"
    MODALITY_TO_ENC_CORE.pop(modality)
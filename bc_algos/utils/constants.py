from strenum import StrEnum


class Modality(StrEnum):
    LOW_DIM="low_dim"
    RGB="rgb"
    DEPTH="depth"


class PolicyType(StrEnum):
    MLP="mlp"
    TRANSFORMER="transformer"
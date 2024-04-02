from strenum import StrEnum


class DatasetType(StrEnum):
    ROBOMIMIC="robomimic"

class Modality(StrEnum):
    LOW_DIM="low_dim"
    RGB="rgb"
    DEPTH="depth"


class PolicyType(StrEnum):
    MLP="mlp"
    TRANSFORMER="transformer"


class RolloutType(StrEnum):
    ROBOMIMIC="robomimic"
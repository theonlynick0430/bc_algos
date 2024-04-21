from strenum import StrEnum


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

class Modality(StrEnum):
    LOW_DIM="low_dim"
    RGB="rgb"
    DEPTH="depth"


class DatasetType(StrEnum):
    ROBOMIMIC="robomimic"
    ISAAC_GYM="isaac_gym"


class GoalMode(StrEnum):
    LAST="last"
    SUBGOAL="subgoal"
    FULL="full"


class PolicyType(StrEnum):
    MLP="mlp"
    TRANSFORMER="transformer"


class EnvType(StrEnum):
    ROBOSUITE="robosuite"
    ISAAC_GYM="isaac_gym"


class RolloutType(StrEnum):
    ROBOMIMIC="robomimic"
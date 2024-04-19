from collections import OrderedDict
import bc_algos.utils.constants as Const
from bc_algos.models.obs_core import LowDimCore, ViTMAECore, ResNet18Core
import numpy as np


"""
Map from modality to encoder core class

Ex: {Modality.LOW_DIM: LowDimCore, Modality.RGB: ResNet18Core}
"""
MODALITY_TO_ENC_CORE_CLASS = OrderedDict()

"""
Map from observation key to encoder core instance

Ex: {"robot0_eef_pos": LowDimCore(), "robot0_eef_quat": LowDimCore(), "agentview_image": ResNet18Core()}
"""
OBS_KEY_TO_ENC_CORE = OrderedDict()

"""
Map from observation key to shape.

Ex: {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "agentview_image": [3, 224, 224]}
"""
OBS_KEY_TO_SHAPE = OrderedDict()

"""
Map from observation key to modality.

Ex: {"robot0_eef_pos": Modality.LOW_DIM, "robot0_eef_quat": Modality.LOW_DIM, "agentview_image": Modality.RGB}
"""
OBS_KEY_TO_MODALITY = OrderedDict()

"""
Map from observation group to observation key.

Ex: {"obs": ["robot0_eef_pos", "robot0_eef_quat"], "goal": ["agentview_image"]}
"""
OBS_GROUP_TO_KEY = OrderedDict()

def register_encoder_core_class(core, modality):
    """
    Register encoder core class for specified modality of data. 

    Args: 
        core (EncoderCore class): encoder core class to register for @modality

        modality (Modality): modality to register encoder core class for
    """
    assert modality not in MODALITY_TO_ENC_CORE_CLASS, f"modality {modality} already registered in MODALITY_TO_ENC_CORE_CLASS"
    MODALITY_TO_ENC_CORE_CLASS[modality] = core

def unregister_encoder_core_class(modality):
    """
    Unregister encoder core class for specified modality of data. 

    Args: 
        modality (Modality): modality to unregister encoder core class for
    """
    assert modality in MODALITY_TO_ENC_CORE_CLASS, f"modality {modality} not registered in MODALITY_TO_ENC_CORE_CLASS"
    MODALITY_TO_ENC_CORE_CLASS.pop(modality)

def register_encoder_core(obs_key, modality, input_shape, **kwargs):
    """
    Register encoder core for specified observation key.

    Args:
        obs_key (str): observation key to register encoder core for

        modality (Modality): modality of @obs_key

        input_shape (array): shape of data corresponding to @obs_key excluding batch and temporal dim

        kwargs (dict): arguments for encoder core
    """
    assert obs_key not in OBS_KEY_TO_ENC_CORE, f"observation key {obs_key} already registered in OBS_KEY_TO_ENC_CORE"
    assert modality in MODALITY_TO_ENC_CORE_CLASS, f"modality {modality} not found in MODALITY_TO_ENC_CORE_CLASS"
    enc_core = MODALITY_TO_ENC_CORE_CLASS[modality](input_shape=input_shape, **kwargs)
    OBS_KEY_TO_ENC_CORE[obs_key] = enc_core

def unregister_encoder_core(obs_key):
    """
    Unregister encoder core for specified observation key.

    Args:
        obs_key (str): observation key to unregister encoder core for
    """
    assert obs_key in OBS_KEY_TO_ENC_CORE, f"observation key {obs_key} not registered in OBS_KEY_TO_ENC_CORE"
    OBS_KEY_TO_ENC_CORE.pop(obs_key)

def init_obs_utils(config):
    """
    Initialize mappings from config.

    Args:
        config (addict): config object
    """
    register_encoder_core_class(core=LowDimCore, modality=Const.Modality.LOW_DIM)
    if config.policy.type == Const.PolicyType.MLP:
        register_encoder_core_class(core=ViTMAECore, modality=Const.Modality.RGB)
    elif config.policy.type == Const.PolicyType.TRANSFORMER:
        register_encoder_core_class(core=ResNet18Core, modality=Const.Modality.RGB)
    for obs_key, shape in config.observation.shapes.items():
        OBS_KEY_TO_SHAPE[obs_key] = shape
    for obs_group, modality_to_obs_key in config.observation.modalities.items():
        for modality, obs_keys in modality_to_obs_key.items():
            for obs_key in obs_keys:
                if obs_group in OBS_GROUP_TO_KEY:
                    OBS_GROUP_TO_KEY[obs_group].append(obs_key)
                else:
                    OBS_GROUP_TO_KEY[obs_group] = [obs_key]
                if obs_key not in OBS_KEY_TO_MODALITY:
                    OBS_KEY_TO_MODALITY[obs_key] = modality
                if obs_key not in OBS_KEY_TO_ENC_CORE:
                    register_encoder_core(
                        obs_key=obs_key,
                        modality=modality, 
                        input_shape=OBS_KEY_TO_SHAPE[obs_key],
                        **config.observation.kwargs[modality]
                    )

def deinit_obs_utils():
    """
    Deinitialize mappings.
    """
    for obs_key in list(OBS_KEY_TO_ENC_CORE.keys()):
        unregister_encoder_core(obs_key=obs_key)
    for modality in list(MODALITY_TO_ENC_CORE_CLASS.keys()):
        unregister_encoder_core_class(modality=modality)

def preprocess_img(img):
    """
    Helper function to preprocess images. Specifically does the following:
    1) Changes shape of @img from [H, W, 3] to [3, H, W]
    2) Changes scale of @img from [0, 255] to [0, 1]

    Args: 
        img (np.array): image data of shape [..., H, W, 3]
    """
    img = np.moveaxis(img.astype(float), -1, -3)
    img /= 255.
    return img.clip(0., 1.)

def compute_traj_stats(traj_dict):
    """
    Compute stats over a trajectory of data. Specifically, compute for each key
    1) mean 
    2) sum of square distances from the mean 

    Args:
        traj_dict (dict): nested dictionary that maps dataset/observation key
            to data of shape [T, ...]

    Returns:
        traj_stats (dict): nested dictionary that maps dataset/observation key
            to a dictionary that contains trajectory stats
    """
    traj_stats = { k : {} for k in traj_dict }
    for k in traj_dict:
        traj_stats[k]["n"] = traj_dict[k].shape[0]
        traj_stats[k]["mean"] = traj_dict[k].mean(axis=0)
        traj_stats[k]["sqdiff"] = ((traj_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0)
    return traj_stats

def aggregate_traj_stats(traj_stats_a, traj_stats_b):
    """
    Aggregate trajectory stats.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    for more information.

    Args:
        traj_stats_a, traj_stats_b (dict): nested dictionary that maps dataset/observation key
            to a dictionary that contains trajectory stats

    Returns:
        merged_stats (dict): nested dictionary that maps dataset/observation key
            to a dictionary that contains merged trajectory stats
    """
    merged_stats = {}
    for k in traj_stats_a:
        n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
        n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
        n = n_a + n_b
        mean = (n_a * avg_a + n_b * avg_b) / n
        delta = (avg_b - avg_a)
        M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
        merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
    return merged_stats

def compute_normalization_stats(traj_stats):
    """
    Compute normalization stats over a trajectory of data. Specifically, compute for each key
    1) mean 
    2) stdv

    Args:
        traj_stats (dict): nested dictionary that maps dataset/observation key
            to a dictionary that contains trajectory stats

    Returns:
        normalization_stats (dict): nested dictionary that maps dataset/observation key
            to a dictionary that contains normalization stats
    """
    normalization_stats = {key: {} for key in traj_stats}
    for key in traj_stats:
        normalization_stats[key]["mean"] = traj_stats[key]["mean"]
        # add small tolerance for stdv
        normalization_stats[key]["stdv"] =  (traj_stats[key]["sqdiff"] / traj_stats[key]["n"] + 1e-3) ** 0.5
    return normalization_stats

def normalize(data, normalization_stats):
    """
    Normalize data according to @normalization_stats.

    Args: 
        data: data to be normalized

        normalization_stats (dict): dictionary that contains normalization stats

    Returns: normalized @data
    """
    return (data - normalization_stats["mean"]) / normalization_stats["stdv"]

def unnormalize(data, normalization_stats):
    """
    Unnormalize data according to @normalization_stats.

    Args: 
        data: data to be unnormalized

        normalization_stats (dict): dictionary that contains normalization stats

    Returns: unnormalized @data
    """
    return data*normalization_stats["stdv"] + normalization_stats["mean"]
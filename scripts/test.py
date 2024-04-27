try:
    from isaacgym import gymtorch
    from isaacgym import gymapi
except ImportError:
    pass
import bc_algos.utils.obs_utils as ObsUtils
from bc_algos.dataset.robomimic import RobomimicDataset
from bc_algos.dataset.isaac_gym import IsaacGymDataset
from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Transformer, MLP
from bc_algos.models.policy_nets import BC_Transformer, BC_MLP
from bc_algos.rollout.robomimic import RobomimicRolloutEnv
from bc_algos.rollout.isaac_gym_simple import IsaacGymSimpleRolloutEnv
import bc_algos.utils.constants as Const
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
import argparse
import json
from addict import Dict
import os
from tqdm import tqdm


def test(config):
    # enforce policy constraints
    if config.policy.type == Const.PolicyType.MLP:
        assert config.dataset.frame_stack == 0, "mlp does not support history"
        assert config.dataset.seq_length == 1, "mlp does not support multi-action output"
    elif config.policy.type == Const.PolicyType.TRANSFORMER:
        assert config.policy.embed_dim%2 == 0, "transformer embedded dim must be even"
    else:
        print(f"unsupported policy type {config.policy.type}")
        exit(1)

    # init obs utils
    ObsUtils.init_obs_utils(config=config)

    # directory handling
    exp_dir = os.path.join(config.experiment.output_dir, config.experiment.name)
    os.makedirs(exp_dir)

    # load datasets and dataloaders
    if config.dataset.type == Const.DatasetType.ROBOMIMIC:
        trainset = RobomimicDataset.factory(config=config, train=True)
        validset = RobomimicDataset.factory(config=config, train=False)
    elif config.dataset.type == Const.DatasetType.ISAAC_GYM:
        trainset = IsaacGymDataset.factory(config=config, train=True)
        validset = IsaacGymDataset.factory(config=config, train=False)    
    else:
        print(f"unsupported dataset type {config.dataset.type}")
        exit(1)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=True)

    # load obs encoder
    obs_group_enc = ObservationGroupEncoder.factory(config=config)

    # load backbone network
    if config.policy.type == Const.PolicyType.MLP:
        backbone = MLP.factory(config=config, embed_dim=obs_group_enc.output_dim)
    elif config.policy.type == Const.PolicyType.TRANSFORMER:
        backbone = Transformer.factory(config=config)

    # load action decoder
    action_dec = ActionDecoder.factory(config=config, input_dim=backbone.output_dim)

    # load policy
    if config.policy.type == Const.PolicyType.MLP:
        policy = BC_MLP(obs_group_enc=obs_group_enc, backbone=backbone, action_dec=action_dec)
    elif config.policy.type == Const.PolicyType.TRANSFORMER:
        policy = BC_Transformer.factory(
            config=config, 
            obs_group_enc=obs_group_enc, 
            backbone=backbone, 
            action_dec=action_dec,
        )

    # load weights
    print(f"loading weights from {config.train.weights} ...")
    policy.load_state_dict(torch.load(config.train.weights))

    # create env for rollout
    if config.rollout.type == Const.RolloutType.ROBOMIMIC:
        rollout_env = RobomimicRolloutEnv.factory(
            config=config, 
            validset=validset,
            policy=policy,
            normalization_stats=trainset.normalization_stats,
        )
    elif config.rollout.type == Const.RolloutType.ISAAC_GYM:
        rollout_env = IsaacGymSimpleRolloutEnv.factory(
            config=config, 
            validset=validset,
            policy=policy,
            normalization_stats=trainset.normalization_stats,
        )
    else:
        print(f"rollout env {config.rollout.type} not supported")
        exit(1)

    accelerator = Accelerator()
    train_loader, valid_loader, policy = accelerator.prepare(
        train_loader, valid_loader, policy
    )

    print("rolling out...")
    with tqdm(total=validset.num_demos, unit='demo') as progress:
        for demo_id in validset.demos:
            _ = rollout_env.rollout_with_stats(
                demo_id=demo_id,
                video_dir=exp_dir,
                device=accelerator.device,
            )
            progress.update(1)

    # deinit obs utils
    ObsUtils.deinit_obs_utils()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to a config json"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to dataset"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="path to saved weights"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output directory"
    )
    args = parser.parse_args()

    assert os.path.exists(args.config), f"config at {args.config} does not exist"
    assert os.path.exists(args.dataset), f"dataset at {args.dataset} does not exist"
    assert os.path.exists(args.weights), f"weights at {args.weights} does not exist"
    assert os.path.exists(args.output), f"output directory at {args.dataset} does not exist"

    # load config 
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)

    exp_dir = os.path.join(args.output, config.experiment.name)
    assert not os.path.exists(exp_dir), f"experiment directory {exp_dir} already exists"

    # overwrite config values
    config.dataset.path = args.dataset
    config.train.weights = args.weights
    config.experiment.output_dir = args.output

    test(config=config)
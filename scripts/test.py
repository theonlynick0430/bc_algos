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
from bc_algos.rollout.isaac_gym import IsaacGymRolloutEnv
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

    # load datasets and dataloaders
    if config.dataset.type == Const.DatasetType.ROBOMIMIC:
        trainset = RobomimicDataset.factory(config=config, train=True)
        validset = RobomimicDataset.factory(
            config=config, 
            train=False, 
            normalization_stats=trainset.normalization_stats,
        )
    elif config.dataset.type == Const.DatasetType.ISAAC_GYM:
        trainset = IsaacGymDataset.factory(config=config, train=True)
        # validset = IsaacGymDataset.factory(
        #     config=config, 
        #     train=False, 
        #     normalization_stats=trainset.normalization_stats,
        # )    
        validset = IsaacGymDataset(
            path="/home/niksrid/nik/bc_algos/datasets/dataset_v5",
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            action_key="actions",
            history=0,
            action_chunk=10,
            pad_history=True,
            pad_action_chunk=True,
            get_pad_mask=True,
            goal_mode="full",
            num_subgoal=25,
            normalize=True,
            normalization_stats=trainset.normalization_stats,
            demo_ids=None,
        ) 
    else:
        print(f"unsupported dataset type {config.dataset.type}")
        exit(1)

    # load obs encoder
    obs_group_enc = ObservationGroupEncoder.factory(config=config)

    # load backbone network
    if config.policy.type == Const.PolicyType.MLP:
        backbone = MLP.factory(config=config, embed_dim=sum(obs_group_enc.output_dim.values()))
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
        rollout_env = IsaacGymRolloutEnv.factory(
            config=config, 
            validset=validset,
            policy=policy,
            normalization_stats=trainset.normalization_stats,
        )
    else:
        print(f"rollout env {config.rollout.type} not supported")
        exit(1)

    accelerator = Accelerator()
    policy = accelerator.prepare(policy)

    print("rolling out...")
    num_pick_success = 0
    num_put_success = 0
    num_success = 0
    with tqdm(total=validset.num_demos, unit='demo') as progress:
        for demo_id in validset.demo_ids:
            results = rollout_env.rollout_with_stats(
                demo_id=demo_id,
                video_dir=config.experiment.output_dir,
                device=accelerator.device,
            )
            num_pick_success += int(results["metrics"]["pick_success"])
            num_put_success += int(results["metrics"]["put_success"])
            num_success += int(results["metrics"]["success"])
            progress.update(1)
    pick_success_rate = num_pick_success / validset.num_demos
    put_success_rate = num_put_success / num_pick_success
    success_rate = num_success / validset.num_demos
    print(f"pick_success_rate: {pick_success_rate}")
    print(f"put_success_rate: {put_success_rate}")
    print(f"overall success_rate: {success_rate}")

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
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # load config 
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)

    # overwrite config values
    config.dataset.path = args.dataset
    config.train.weights = args.weights
    config.experiment.output_dir = args.output

    test(config=config)
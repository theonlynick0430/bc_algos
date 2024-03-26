import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.train_utils as TrainUtils
from bc_algos.models.obs_core import EncoderCore, VisualCore
from bc_algos.dataset.dataset import DatasetType
from bc_algos.dataset.robomimic import RobomimicDataset
from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Transformer, MLP
from bc_algos.models.policy_nets import BC_Transformer, BC_MLP, PolicyType
from bc_algos.rollout.rollout_env import RolloutType
from bc_algos.rollout.robomimic import RobomimicRolloutEnv
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from accelerate import Accelerator
import wandb
import argparse
import json
from addict import Dict
import os
from tqdm import tqdm


def train(config):
    # enforce policy constraints
    if config.policy.type == PolicyType.MLP:
        assert config.dataset.frame_stack == 0, "mlp does not support history"
        assert config.dataset.seq_length == 1, "mlp does not support multi-action output"
    elif config.policy.type == PolicyType.TRANSFORMER:
        assert config.dataset.frame_stack+1 == config.dataset.seq_length, \
            "history must = # predicted actions for transformer"
    else:
        print(f"unsupported policy type {config.policy.type}")
        exit(1)

    # register encoder cores
    ObsUtils.register_encoder_core(EncoderCore, ObsUtils.Modality.LOW_DIM)
    ObsUtils.register_encoder_core(VisualCore, ObsUtils.Modality.RGB)

    # init obs utils
    ObsUtils.init_obs_utils(config=config)

    # directory handling
    exp_dir = os.path.join(config.experiment.output_dir, config.experiment.name)
    os.makedirs(exp_dir)
    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir)
    rollout_dir = os.path.join(exp_dir, "rollout")
    os.makedirs(rollout_dir)

    # load datasets and dataloaders
    if config.dataset.type == DatasetType.ROBOMIMIC:
        trainset = RobomimicDataset.factory(config=config, train=True)
        validset = RobomimicDataset.factory(config=config, train=False)
    else:
        print(f"unsupported dataset type {config.dataset.type}")
        exit(1)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=True)

    # load obs encoder
    obs_group_enc = ObservationGroupEncoder.factory(config=config)

    # load backbone network
    if config.policy.type == PolicyType.MLP:
        backbone = MLP.factory(config=config, input_dim=obs_group_enc.output_dim)
    elif config.policy.type == PolicyType.TRANSFORMER:
        backbone = Transformer.factory(config=config, input_dim=obs_group_enc.output_dim)

    # load action decoder
    act_dec = ActionDecoder.factory(config=config, input_dim=backbone.output_dim)

    # load policy
    if config.policy.type == PolicyType.MLP:
        policy = BC_MLP(obs_group_enc, backbone, act_dec)
    elif config.policy.type == PolicyType.TRANSFORMER:
        policy = BC_Transformer(obs_group_enc, backbone, act_dec)

    # create optimizer and loss function
    optimizer = optim.Adam(policy.parameters(), lr=config.train.learning_rate)
    if config.train.loss == "L2":
        loss_fn = nn.MSELoss()
    elif config.train.loss == "L1":
        loss_fn = nn.L1Loss()
    else:
        print(f"loss type {config.train.loss} not supported")
        exit(1)

    # create env for rollout
    if config.rollout.type == RolloutType.ROBOMIMIC:
        rollout_env = RobomimicRolloutEnv.factory(config=config, validset=validset)
    else:
        print(f"rollout env {config.rollout.type} not supported")
        exit(1)

    accelerator = Accelerator()
    train_loader, valid_loader, policy, optimizer = accelerator.prepare(
        train_loader, valid_loader, policy, optimizer
    )

    # wandb login
    wandb.init(project=config.experiment.name)

    # iterate epochs
    valid_ct = 1
    rollout_ct = 1
    save_ct = 1
    for epoch in range(config.train.epochs):
        print(f"epoch {epoch}")

        # TRAINING
        print("training...")
        TrainUtils.run_epoch(
            model=policy,
            data_loader=train_loader,
            loss_fn=loss_fn,
            frame_stack=config.dataset.frame_stack,
            optimizer=optimizer,
            validate=False,
            device=accelerator.device,
        )

        # VALIDATION
        if valid_ct == config.experiment.valid_rate:
            print("validating...")
            TrainUtils.run_epoch(
                model=policy,
                data_loader=valid_loader,
                loss_fn=loss_fn,
                frame_stack=config.dataset.frame_stack,
                optimizer=optimizer,
                validate=True,
                device=accelerator.device,
            )
            valid_ct = 1

        # ROLLOUT
        if rollout_ct == config.experiment.rollout_rate:
            print("rolling out...")
            rollout_epoch_dir = os.path.join(rollout_dir, f"{epoch}")
            os.mkdir(rollout_epoch_dir)
            with tqdm(total=validset.num_demos, unit='demo') as progress:
                for demo in validset.demos:
                    _ = rollout_env.rollout_with_stats(
                        policy=policy,
                        demo_id=demo,
                        video_dir=rollout_epoch_dir,
                        device=accelerator.device,
                    )
                    progress.update(1)
            rollout_ct = 1

        # SAVE WEIGHTS
        if save_ct == config.experiment.save_rate:
            print("saving weights...")
            torch.save(policy.state_dict(), os.path.join(weights_dir, f"model_{epoch}.pth"))
            save_ct = 1

    # save final model
    torch.save(policy.state_dict(), os.path.join(weights_dir, f"model.pth"))

    # unregister encoder cores
    ObsUtils.unregister_encoder_core(ObsUtils.Modality.LOW_DIM)
    ObsUtils.unregister_encoder_core(ObsUtils.Modality.RGB)


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
        help="path to dataset",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output directory",
    )
    args = parser.parse_args()

    assert os.path.exists(args.config), f"config at {args.config} does not exist"
    assert os.path.exists(args.dataset), f"dataset at {args.dataset} does not exist"
    assert os.path.exists(args.output), f"output directory at {args.dataset} does not exist"

    # load config 
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)

    exp_dir = os.path.join(args.output, config.experiment.name)
    assert not os.path.exists(exp_dir), f"experiment directory {exp_dir} already exists"

    # overwrite config values
    config.experiment.output_dir = args.output
    config.dataset.path = args.dataset

    train(config=config)
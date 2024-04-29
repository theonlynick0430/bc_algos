from bc_algos.envs.isaac_gym_simple import IsaacGymEnvSimple
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.constants as Const
from bc_algos.utils.misc import load_gzip_pickle, save_gzip_pickle
from addict import Dict
import os
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np

    
def preprocess_dataset(
    dataset_path, 
    output_path, 
    dataset_keys, 
    obs_keys, 
    normalization_stats=None, 
):
    num_demos = int(len(os.listdir(dataset_path)))
    with tqdm(total=num_demos, desc="preprocessing dataset", unit='demo') as progress_bar:
        for sub_path in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, sub_path)
            run = load_gzip_pickle(filename=run_path)

            for obs_key in obs_keys:
                if ObsUtils.OBS_KEY_TO_MODALITY[obs_key] == Const.Modality.RGB:
                    run["obs"][obs_key] = IsaacGymEnvSimple.preprocess_img(img=run["obs"][obs_key])

            if normalization_stats is not None:
                for dataset_key in dataset_keys:
                    if dataset_key in normalization_stats:
                        run["policy"][dataset_key] = ObsUtils.normalize(
                            data=run["policy"][dataset_key], 
                            normalization_stats=normalization_stats[dataset_key], 
                        )
                for obs_key in obs_keys:
                    if obs_key in normalization_stats:
                        run["obs"][obs_key] = ObsUtils.normalize(
                            data=run["obs"][obs_key], 
                            normalization_stats=normalization_stats[obs_key], 
                        )

            new_run_path = os.path.join(output_path, sub_path)
            save_gzip_pickle(data=run, filename=new_run_path)

            progress_bar.update(1)

    # create train, val splits
    split_idx = int(num_demos*0.9)
    train_demo_id = np.arange(split_idx, dtype=int)
    valid_demo_id = np.arange(split_idx, num_demos, dtype=int)
    split = {"train": train_demo_id, "valid": valid_demo_id}
    split_path = os.path.join(output_path, "split.pkl.gzip")
    save_gzip_pickle(data=split, filename=split_path)

def compute_normalization_stats(dataset_path, dataset_keys, obs_keys, device=None):
    traj_dict = {}
    merged_stats = {}

    num_demos = int(len(os.listdir(dataset_path)))
    with tqdm(total=num_demos, desc="computing normalization stats", unit="demo") as progress_bar:
        for i, sub_path in enumerate(os.listdir(dataset_path)):
            run_path = os.path.join(dataset_path, sub_path)
            run = load_gzip_pickle(filename=run_path)

            traj_dict = {obs_key: torch.from_numpy(run["obs"][obs_key]).to(device) for obs_key in obs_keys}
            for dataset_key in dataset_keys:
                traj_dict[dataset_key] = torch.from_numpy(run["policy"][dataset_key]).to(device)

            if i == 0:
                merged_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
            else:
                traj_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                merged_stats = ObsUtils.aggregate_traj_stats(traj_stats_a=merged_stats, traj_stats_b=traj_stats)

            progress_bar.update(1)
    
    return TensorUtils.to_numpy(ObsUtils.compute_normalization_stats(traj_stats=merged_stats, tol=1e-3))

def main(args):
    assert os.path.exists(args.config), f"config at {args.config} does not exist"
    assert os.path.exists(args.dataset), f"dataset at {args.dataset} does not exist"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = None
    if args.cuda is True:
        device = torch.cuda.device(0)

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)
    obs_keys = list(config.observation.shapes.keys())
    dataset_keys = config.dataset.dataset_keys

    ObsUtils.init_obs_utils(config=config)

    normalization_stats = None
    if args.normalize:
        normalization_stats = compute_normalization_stats(
            dataset_path=args.dataset, 
            dataset_keys=dataset_keys, 
            # don't compute normalization stats for RGB data since we use backbone encoders
            # with their own normalization stats
            obs_keys=[obs_key for obs_key in obs_keys if ObsUtils.OBS_KEY_TO_MODALITY[obs_key] != Const.Modality.RGB], 
            device=device,
        )
        stats_path = os.path.join(args.output, "normalization_stats.pkl.gzip")
        save_gzip_pickle(data=normalization_stats, filename=stats_path)

    preprocess_dataset(
        dataset_path=args.dataset,
        output_path=args.output, 
        obs_keys=obs_keys,
        normalization_stats=normalization_stats,
    )


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
        help="path to dataset directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output directory"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
    )

    parser.add_argument(
        "--world",
        action="store_true",
    )

    parser.add_argument(
        "--othor6D",
        action="store_true",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)

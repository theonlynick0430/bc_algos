"""
Script for preprocessing Isaac Gym dataset for training.
"""
from bc_algos.envs.isaac_gym_simple import IsaacGymEnvSimple
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
from bc_algos.utils.misc import load_gzip_pickle, save_gzip_pickle
from addict import Dict
import os
import argparse
import json
from tqdm import tqdm
import numpy as np

    
def preprocess_dataset(config, dataset_path, output_path):
    # init obs utils
    ObsUtils.init_obs_utils(config=config)

    num_demos = int(len(os.listdir(dataset_path)))
    obs_keys = list(ObsUtils.OBS_KEY_TO_MODALITY.keys())

    with tqdm(total=num_demos, desc="preprocessing dataset", unit='demo') as progress_bar:
        for sub_path in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, sub_path)
            run = load_gzip_pickle(filename=run_path)

            for obs_key in obs_keys:
                if ObsUtils.OBS_KEY_TO_MODALITY[obs_key] == Const.Modality.RGB:
                    run["obs"][obs_key] = IsaacGymEnvSimple.preprocess_img(img=run["obs"][obs_key])

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
    args = parser.parse_args()

    assert os.path.exists(args.config), f"config at {args.config} does not exist"
    assert os.path.exists(args.dataset), f"dataset at {args.dataset} does not exist"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # load config 
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)

    preprocess_dataset(config=config, dataset_path=args.dataset, output_path=args.output)

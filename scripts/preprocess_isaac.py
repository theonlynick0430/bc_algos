from bc_algos.envs.isaac_gym import IsaacGymEnv
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
from bc_algos.utils.misc import load_gzip_pickle
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, matrix_to_axis_angle
import torchvision.transforms as T
from addict import Dict
import os
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
import pickle

    
def preprocess_dataset(
    dataset_path, 
    output_path, 
    obs_keys,
    action_key,
    rot=False,
    device=None,
):
    num_demos = int(len(os.listdir(dataset_path)))
    with tqdm(total=num_demos, desc="preprocessing dataset", unit='demo') as progress_bar:
        for sub_path in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, sub_path)
            run = load_gzip_pickle(filename=run_path)

            demo = {}
            demo["obs"] = {obs_key: run["obs"][obs_key] for obs_key in obs_keys}
            demo["policy"] = {}
            demo["policy"][action_key] = run["policy"][action_key]
            demo = TensorUtils.to_tensor(x=demo, device=device)

            # preprocess images
            trans = T.Resize((256,384))
            for obs_key in obs_keys:
                if ObsUtils.OBS_KEY_TO_MODALITY[obs_key] == Const.Modality.RGB:
                    img = demo["obs"][obs_key]
                    img = IsaacGymEnv.preprocess_img(img=img)
                    img = trans(img)
                    demo["obs"][obs_key] = img.clip(0., 1.)

            # preprocess gripper actions to be more conducive for learning
            mask = demo["policy"][action_key][:, -1] >= 0
            demo["policy"][action_key][:, -1][mask] = 1
            demo["policy"][action_key][:, -1][~mask] = -1

            if rot:
                # convert orientation to ortho6D and world frame
                state_pos = demo["obs"]["robot0_eef_pos"]
                state_quat = demo["obs"]["robot0_eef_quat"]
                state_mat = quaternion_to_matrix(state_quat)
                state_ortho6D = matrix_to_rotation_6d(state_mat)
                ee_pose = TensorUtils.se3_matrix(rot=state_mat, pos=state_pos)
                
                action_pos = demo["policy"][action_key][:, :3]
                action_aa = demo["policy"][action_key][:, 3:-1]
                action_grip = demo["policy"][action_key][:, -1:]
                action_mat = axis_angle_to_matrix(action_aa)
                action_ortho6D = matrix_to_rotation_6d(action_mat)
                action_pose = TensorUtils.se3_matrix(rot=action_mat, pos=action_pos)
                
                action_pose_world = TensorUtils.change_basis(pose=action_pose, transform=ee_pose, standard=False)
                action_pos_world = action_pose_world[:, :3, 3]
                action_mat_world = action_pose_world[:, :3, :3]
                action_aa_world = matrix_to_axis_angle(action_mat_world)
                action_ortho6D_world = matrix_to_rotation_6d(action_mat_world)

                demo["obs"]["robot0_eef_ortho6D"] = state_ortho6D
                demo["policy"][action_key+"_ortho6D"] = torch.cat((action_pos, action_ortho6D, action_grip), dim=-1)
                demo["policy"][action_key+"_world"] =  torch.cat((action_pos_world, action_aa_world, action_grip), dim=-1)
                demo["policy"][action_key+"_ortho6D_world"] = torch.cat((action_pos_world, action_ortho6D_world, action_grip), dim=-1)

            demo = TensorUtils.to_numpy(x=demo)

            # for rollout
            demo["obs"]["cubes_pos"] = run["obs"]["cubes_pos"]
            demo["obs"]["cubes_quat"] = run["obs"]["cubes_quat"]
            demo["obs"]["q"] = run["obs"]["q"]
            demo["metadata"] = run["metadata"]

            new_run_path = os.path.join(output_path, sub_path.split('.')[0] + ".pickle")
            with open(new_run_path, 'wb') as handle:
                pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

            progress_bar.update(1)

    # create train, val splits
    split_idx = int(num_demos*0.9)
    train_demo_id = np.arange(split_idx, dtype=int)
    valid_demo_id = np.arange(split_idx, num_demos, dtype=int)
    split = {"train": train_demo_id, "valid": valid_demo_id}
    split_path = os.path.join(output_path, "split.pickle")
    with open(split_path, 'wb') as handle:
        pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    assert os.path.exists(args.config), f"config at {args.config} does not exist"
    assert os.path.exists(args.dataset), f"dataset at {args.dataset} does not exist"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = None
    if args.cuda is True:
        device = torch.device(0)

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = Dict(config)
    obs_keys = list(config.observation.shapes.keys())
    action_key = config.dataset.action_key

    ObsUtils.init_obs_utils(config=config)

    preprocess_dataset(
        dataset_path=args.dataset,
        output_path=args.output, 
        obs_keys=obs_keys,
        action_key=action_key,
        rot=args.rot,
        device=device,
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
        "--rot",
        action="store_true",
        help="flag to convert rotations to ortho6D and world frame"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)

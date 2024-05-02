from bc_algos.envs.isaac_gym_simple import IsaacGymEnvSimple
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
from bc_algos.utils.misc import load_gzip_pickle, save_gzip_pickle
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
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
    obs_keys,
    action_key,
    use_ortho6D=False,
    use_world=False,
    device=None,
):
    num_demos = int(len(os.listdir(dataset_path)))
    with tqdm(total=num_demos, desc="preprocessing dataset", unit='demo') as progress_bar:
        for sub_path in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, sub_path)
            run = load_gzip_pickle(filename=run_path)

            demo = {}
            obs_keys = obs_keys + ["cubes_pos", "cubes_quat", "q"]
            demo["obs"] = {obs_key: run["obs"][obs_key] for obs_key in obs_keys}
            demo["policy"] = {}
            demo["policy"][action_key] = run["policy"][action_key]
            demo = TensorUtils.to_tensor(x=demo, device=device)

            # preprocess images
            for obs_key in obs_keys:
                if ObsUtils.OBS_KEY_TO_MODALITY[obs_key] == Const.Modality.RGB:
                    demo["obs"][obs_key] = IsaacGymEnvSimple.preprocess_img(img=demo["obs"][obs_key])
            
            # convert orientation to ortho6D / world frame
            if use_ortho6D or use_world:
                state_quat = demo["obs"]["robot0_eef_quat"]
                state_mat = quaternion_to_matrix(state_quat)
                if use_ortho6D:
                    state_ortho6D = matrix_to_rotation_6d(state_mat)
                    demo["obs"]["robot0_eef_ortho6D"] = state_ortho6D
                action_pos = demo["policy"][action_key][:, :3]
                action_aa = demo["policy"][action_key][:, 3:-1]
                action_grip = demo["policy"][action_key][:, -1:]
                action_mat = axis_angle_to_matrix(action_aa)
                if use_world:
                    state_pos = demo["obs"]["robot0_eef_pos"]
                    ee_pose = TensorUtils.se3_matrix(rot=state_mat, pos=state_pos)
                    action_pose = TensorUtils.se3_matrix(rot=action_mat, pos=action_pos)
                    action_pose = TensorUtils.change_basis(pose=action_pose, transform=ee_pose, standard=False)
                    action_pos = action_pose[:, :3, 3]
                    action_mat = action_pose[:, :3, :3]
                if use_ortho6D:
                    action_ortho6D = matrix_to_rotation_6d(action_mat)
                    action = torch.cat((action_pos, action_ortho6D, action_grip), dim=-1)
                    demo["policy"][action_key+"_ortho6D_world" if use_world else "_ortho6D"] = action
                else:
                    # hack since matrix_to_axis_angle is broken
                    action_aa = quaternion_to_axis_angle(matrix_to_quaternion(action_mat))
                    action = torch.cat((action_pos, action_aa, action_grip), dim=-1)
                    demo["policy"][action_key+"_world"] = action

            demo = TensorUtils.to_numpy(x=demo)
            demo["metadata"] = run["metadata"]
            # run.update(demo)

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
        use_ortho6D=args.ortho6D,
        use_world=args.world,
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
        "--ortho6D",
        action="store_true",
    )

    parser.add_argument(
        "--world",
        action="store_true",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)

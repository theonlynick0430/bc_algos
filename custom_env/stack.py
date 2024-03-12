import bc_benchmark_algos.custom_env.utils as RobosuiteEnvUtils
import numpy as np
import argparse
import json
import time
import os
import shutil
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5


# x, y, z -> out of screen, x-axis, y-axis
def collect_demos(env, demos, render=True):
    i = 0
    while i < demos:
        # create new eps
        env.reset()
        obs = env._get_observations(force_update=True)
        # navigate over small block and close gripper
        target_state = RobosuiteEnvUtils.get_current_state(obs=obs)
        target_state[0] = obs["cubeA_pos"]
        target_state[0][2] += 0.05
        obs, _ = RobosuiteEnvUtils.linear_action(env=env, target_state=target_state, gripper_cmd=RobosuiteEnvUtils.GRIPPER_CLOSED, thresh=0.025, max_steps=100, render=render)
        # open gripper
        obs = RobosuiteEnvUtils.gripper_action(env=env, grasp=False, render=render)
        # grasp
        target_state = RobosuiteEnvUtils.get_current_state(obs=obs)
        target_state[0][2] -= 0.075
        obs, _ = RobosuiteEnvUtils.linear_action(env=env, target_state=target_state, gripper_cmd=RobosuiteEnvUtils.GRIPPER_OPEN, thresh=0.025, max_steps=100, render=render)
        obs = RobosuiteEnvUtils.gripper_action(env=env, grasp=True, render=render)
        # lift up 
        target_state = RobosuiteEnvUtils.get_current_state(obs=obs)
        target_state[0][2] += 0.1
        obs, _ = RobosuiteEnvUtils.linear_action(env=env, target_state=target_state, gripper_cmd=RobosuiteEnvUtils.GRIPPER_CLOSED, thresh=0.025, max_steps=100, render=render)
        # navigate over big block
        target_state = RobosuiteEnvUtils.get_current_state(obs=obs)
        target_state[0] =  obs["cubeB_pos"]
        target_state[0][2] += 0.05
        obs, _ = RobosuiteEnvUtils.linear_action(env=env, target_state=target_state, gripper_cmd=RobosuiteEnvUtils.GRIPPER_CLOSED, thresh=0.025, max_steps=100, render=render)
        # check if dropped
        # if np.linalg.norm(obs["gripper_to_cubeA"]) > 0.025:
        # drop 
        RobosuiteEnvUtils.gripper_action(env=env, grasp=False, render=render)
        if env._check_success():
            # success
            print(f"finished collecting demo {i}")
            i += 1
    env.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    config, env = RobosuiteEnvUtils.get_env(env_name="Stack", render=args.render)
    env_info = json.dumps(config)
    tmp_dir = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_dir)
    env = VisualizationWrapper(env)
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    collect_demos(env=env, demos=args.demos, render=args.render)
    gather_demonstrations_as_hdf5(tmp_dir, new_dir, env_info)
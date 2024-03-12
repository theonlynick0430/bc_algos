
from robosuite.utils.input_utils import *
from robosuite.utils.transform_utils import quat2mat, mat2quat, quat2axisangle
import numpy as np
import robosuite as suite


GRIPPER_CLOSED = 1
GRIPPER_OPEN = -1

def get_env(env_name, render=True):
    config = {
        "env_name": env_name,
        "robots": ["Panda"],
        "controller_configs": suite.load_controller_config(default_controller="OSC_POSE"),
    }
    return config, suite.make(
        **config,
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True, 
        control_freq=20,
    )

def get_current_state(obs):
    ee_pos = obs["robot0_eef_pos"].copy()
    ee_quat = obs["robot0_eef_quat"].copy()
    return [ee_pos, ee_quat]

def linear_action(env, target_state, update_target_state=None, gripper_cmd=0, thresh=0.05, max_steps=25, render=True):
    obs = env._get_observations(force_update=True)
    state = get_current_state(obs)
    error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    steps = 0
    while error > thresh:
        dmat = np.dot(quat2mat(target_state[1]), quat2mat(state[1]).T)
        action = np.concatenate((target_state[0]-state[0], quat2axisangle(mat2quat(dmat)), [gripper_cmd]))
        action[:3] /= np.linalg.norm(action[:3])
        obs, _, _, _ = env.step(action) 
        steps += 1
        if steps >= max_steps:
            return obs, False
        if render:
            env.render()
        state = get_current_state(obs)
        if update_target_state:
            target_state = update_target_state(obs, target_state)
        error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    return obs, True

def gripper_action(env, grasp=True, render=True):
    action = np.zeros(7)
    if grasp:
        action[6] = GRIPPER_CLOSED
    else:
        action[6] = GRIPPER_OPEN
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    return obs

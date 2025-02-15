
from robosuite.utils.input_utils import *
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply
import robosuite as suite
import numpy as np 

CUBE_POS_KEY = "cube_pos"
CUBE_QUAT_KEY = "cube_quat"
ROBOT_POS_KEY = "robot0_eef_pos"
ROBOT_QUAT_KEY = "robot0_eef_quat"
IMAGE_OBS_KEY = "agentview_image"

def get_env():
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    controller_config["output_max"] = np.ones(6)
    controller_config["output_min"] = -np.ones(6)
    return suite.make(
        env_name="Lift",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        control_freq=20, 
    )

def encode_state(pos, quat):
    return np.concatenate((pos, quat))

def decode_obs(obs):
    pos = obs[ROBOT_POS_KEY]
    quat = obs[ROBOT_QUAT_KEY]
    return pos, quat

def inverse_quaternion(quaternion):
    x, y, z, w = quaternion
    return np.array([-x, -y, -z, w])

def linear_action(env, target_pos, target_quat, trans_step=0.025, rot_step=0.1, thresh=0.05, render=True):
    target = encode_state(target_pos, target_quat)
    obs = env._get_observations()
    pos, quat = decode_obs(obs)
    error = np.linalg.norm(target-encode_state(pos, quat))
    while error > thresh:
        action_pos = target_pos-pos
        action_pos = action_pos
        action_pos_norm = np.linalg.norm(action_pos)
        if action_pos_norm > trans_step:
            action_pos = action_pos/(action_pos_norm+1e-5)*trans_step
        # for orientation, env takes in delta axis-angle commands relative to world axis
        inv_quat = inverse_quaternion(quat)
        action_quat = quat_multiply(target_quat, inv_quat)
        action_aa = quat2axisangle(action_quat)
        action_aa = action_aa
        action_aa_norm = np.linalg.norm(action_aa)
        if action_aa_norm > trans_step:
            action_aa = action_aa/(action_aa_norm+1e-5)*rot_step
        action = np.concatenate((action_pos, action_aa, [0]))
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
        pos, quat = decode_obs(obs)
        error = np.linalg.norm(target-encode_state(pos, quat))
    return obs, True

def gripper_close(env, render=True):
    obs = None
    for _ in range(10):
        action = np.zeros(7)
        action[-1] = 1
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    return obs

def gripper_open(env, render=True):
    obs = None
    for _ in range(10):
        action = np.zeros(7)
        action[-1] = -1
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    return obs

def signed_smallest_angle_from_axis(theta):
    phi = theta
    if phi < 0: 
        phi += 2*np.pi
    if phi >= 0 and phi < np.pi/2: 
        if phi >= np.pi/4:
            return np.pi/2 - phi
        else:
            return -phi
    elif phi >= np.pi/2 and phi < np.pi:
        if phi >= 3*np.pi/4:
            return np.pi - phi
        else:
            return np.pi/2 - phi
    elif phi >= np.pi and phi < 3*np.pi/2:
        if phi >= 5*np.pi/4:
            return 3*np.pi/2 - phi
        else:
            return np.pi - phi
    else:
        if phi >= 7*np.pi/4:
            return 2*np.pi - phi
        else:
            return 3*np.pi/2 - phi
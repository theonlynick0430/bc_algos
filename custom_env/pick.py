from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, axisangle2quat
from robosuite.utils.binding_utils import MjSimState
from custom_env.utils import get_env, linear_action, gripper_open, gripper_close, signed_smallest_angle_from_axis
from custom_env.utils import CUBE_POS_KEY, CUBE_QUAT_KEY, ROBOT_QUAT_KEY
from custom_env.env_wrapper import DataCollectionWrapper
import numpy as np
import argparse
import h5py


def collect_eps(env):
    obs = env.reset()

    # move to correct pos
    target_pos = obs[CUBE_POS_KEY]
    target_pos[-1] += 0.05
    obs, _ = linear_action(env, target_pos, obs[ROBOT_QUAT_KEY])

    # move to correct orientation
    cube_quat = obs[CUBE_QUAT_KEY]
    cube_aa = quat2axisangle(cube_quat)
    z_rot_angle = np.linalg.norm(cube_aa)
    # print(f"ANGLE: {z_rot_angle}")
    z_rot_angle = signed_smallest_angle_from_axis(z_rot_angle)
    # print(f"SHORTEST ANGLE: {z_rot_angle}")
    cube_quat = axisangle2quat(np.array([0., 0., z_rot_angle]))
    target_quat = quat_multiply(obs[ROBOT_QUAT_KEY], cube_quat)
    obs, _ = linear_action(env, target_pos, target_quat)

    # open gripper
    obs = gripper_open(env)

    # move down
    target_pos = obs[CUBE_POS_KEY]
    target_pos[-1] -= 0.05
    obs, _ = linear_action(env, target_pos, target_quat)

    # close gripper
    obs = gripper_close(env)

    # move up 
    target_pos[-1] += 0.1
    obs, _ = linear_action(env, target_pos, target_quat)

    # save data 
    env.flush()

def playback_demo(env, filepath, eps):
    with h5py.File(filepath, 'r', swmr=True) as f:
        env.reset()

        demo = f[f"demo_{eps}"]
        q_init = demo["q"][0][:]
        qdot_init = demo["qdot"][0][:]
        sim_state = MjSimState(0, q_init, qdot_init)
        env.sim.set_state(sim_state)
        env.sim.forward()
        # small error found when setting q directly so move to inital pose
        action = demo["action"][:]
        for i in range(len(action)):
            env.step(action[i])
            env.render()
        # ee_pos = demo["ee_pos"][:]
        # ee_quat = demo["ee_quat"][:]
        # linear_action(env, ee_pos[0], ee_quat[0], render=False, thresh=0.01)

        # for i in range(len(ee_pos)):
        #     linear_action(env, ee_pos[i], ee_quat[i], thresh=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--playback", action="store_true")
    parser.add_argument("--eps", type=int) # if playback, the eps to playback, otherwise, the number of eps to collect
    args = parser.parse_args()

    env = get_env()
    if args.playback:
        playback_demo(env, args.filepath, args.eps)
    else:
        env = DataCollectionWrapper(env, args.filepath)
        env.record()
        for i in range(args.eps):
            collect_eps(env)
            print(f"collected data for eps: {i}")
        env.stop_record()
    env.close()
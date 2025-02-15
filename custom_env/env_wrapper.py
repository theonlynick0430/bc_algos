import numpy as np
from robosuite.wrappers import Wrapper
from custom_env.utils import ROBOT_POS_KEY, ROBOT_QUAT_KEY, IMAGE_OBS_KEY
import h5py


class DataCollectionWrapper(Wrapper):

    def __init__(self, env, file_path):
        super().__init__(env)

        self.file_path = file_path
        self.demo = 0
        self._record = False
        self.obs = None

    def _reset_data(self):
        # clear data to be saved on disk  
        self.q = []
        self.qdot = []
        self.ee_pos = []
        self.ee_quat = []
        self.img_obs = []
        self.action = []  

    def reset(self):
        obs = super().reset()
        self.obs = obs
        self._reset_data()
        return obs
    
    def record(self):
        self._record = True

    def stop_record(self):
        self._record = False

    def step(self, action):
        # save data 
        if self._record:
            state = self.sim.get_state()
            self.q.append(state.qpos)
            self.qdot.append(state.qvel)
            self.ee_pos.append(self.obs[ROBOT_POS_KEY])
            self.ee_quat.append(self.obs[ROBOT_QUAT_KEY])
            self.img_obs.append(self.obs[IMAGE_OBS_KEY])
            self.action.append(action)

        self.obs, reward, complete, misc = super().step(action)
        return self.obs, reward, complete, misc

    def flush(self):
        """
        Method to flush internal state to disk after episode has ended.
        It is the user's responsibilty to save data to disk before ending programs.
        """
        with h5py.File(self.file_path, 'a', libver='latest') as f:
            f.swmr_mode = True # enable single writer, multiple readers
            demo = f.create_group(f"demo_{self.demo}")
            demo["q"] = np.array(self.q)
            demo["qdot"] = np.array(self.qdot)
            demo["ee_pos"] = np.array(self.ee_pos)
            demo["ee_quat"] = np.array(self.ee_quat)
            # by default images are reflected across x-axis
            demo["img_obs"] = np.transpose(np.flip(np.array(self.img_obs), axis=1), axes=(0, 3, 1, 2))
            demo["action"] = np.array(self.action)

        self.demo += 1

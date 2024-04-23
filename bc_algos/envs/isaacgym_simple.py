import numpy as np
import omegaconf

import isaacgymenvs
from isaacgymenvs.tasks import MentalModelsTaskSimple

import torch

from bc_algos.envs.env_base import EnvBase


class IsaacGymEnvSimple(EnvBase):

    @classmethod
    def preprocess_img(cls, img):
        """
        Helper function to preprocess images from Isaac Gym environment.
        Specifically does the following:
        1) Removes alpha (last) channel from @img.
        2) Changes shape of @img from [H, W, 3] to [3, H, W]
        3) Changes scale of @img from [0, 255] to [0, 1]

        Args:
            img (np.array): image data of shape [..., H, W, 4]

        Returns: preprocessed @img of shape [..., 3, H, W].
        """
        img = img[:, :, :-1]
        img = np.moveaxis(img.astype(float), -1, -3)
        img /= 255.
        return img.clip(0., 1.)

    def __init__(self,
                 env_name,
                 obs_key_to_modality,
                 render=False,
                 use_image_obs=False,
                 use_depth_obs=False,
                 cfg: dict = None,
                 **kwargs
                 ):
        super().__init__(env_name, obs_key_to_modality, render, use_image_obs, use_depth_obs)
        self.cfg = omegaconf.OmegaConf.create(cfg)

        self.env: MentalModelsTaskSimple = isaacgymenvs.make(
            self.cfg.seed,
            self.cfg.task_name,
            1,  # Use single environment.
            self.cfg.sim_device,
            self.cfg.rl_device,
            self.cfg.graphics_device_id,
            self.cfg.headless,
            self.cfg.multi_gpu,
            self.cfg.capture_video,
            self.cfg.force_render,
            self.cfg,
        )
        self.init_cycles = 10
        self.device = self.env.device
        self.env_id = torch.tensor([0], dtype=torch.long, device=self.device)

    def load_env(self, xml):
        # Not used in this environment.
        pass

    def step(self, action):
        action = torch.tensor(action, device=self.device).float().unsqueeze(0)
        return self.env.step(action)

    def reset(self):
        # Resets randomly.
        self.env.reset_idx(self.env_id)

        # "Warm up" the environment.
        for _ in range(self.init_cycles):
            self.step(np.zeros(7))

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        For Isaac Gym, state is metadata dictionary.
        """
        assert type(state) == dict, "State must be a dictionary."
        block_colors = state["block_colors"]
        block_init_pose = state["block_init_pose"]
        q_init = torch.from_numpy(state["start_q"]).to(self.device).float().unsqueeze(0)
        block_init_pose = torch.from_numpy(block_init_pose).to(self.device).float().unsqueeze(0)
        self.env.reset_idx(self.env_id, block_colors, block_init_pose)

        # self.env._refresh()
        # obs_dict = self.get_observation()
        # print("q: " + str(obs_dict["obs"]["q"]))

        # "Warm up" the environment.
        for _ in range(self.init_cycles):
            self.env.set_arm_dof(self.env_id, q_init)
            obs_dict = self.step(np.zeros(7))
            # print("q: " + str(obs_dict["obs"]["q"]))

        obs_dict = self.get_observation()
        # print("des q: " + str(q_init.cpu().numpy()[0]))
        # print("q: " + str(obs_dict["obs"]["q"]))

    def render(self, height=None, width=None, camera_name=None, on_screen=False):
        obs_dict = self.env.get_observations()
        return obs_dict["obs"]["agentview_image"][0].cpu().numpy()

    def get_observation(self):
        obs_dict = self.env.get_observations()
        return obs_dict

    def is_success(self):
        return False

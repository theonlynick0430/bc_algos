import torch

import isaacgymenvs
from bc_algos.envs.env_base import EnvBase
from isaacgymenvs.tasks import MentalModelsTaskSimple


class EnvIsaacGymSimple(EnvBase):

    def __init__(self,
                 env_name,
                 obs_key_to_modality,
                 render=False,
                 use_image_obs=False,
                 use_depth_obs=False,
                 cfg=None,
                 **kwargs
                 ):
        super().__init__(env_name, obs_key_to_modality, render, use_image_obs, use_depth_obs)
        self.cfg = cfg

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
        self.device = self.env.device
        self.env_id = torch.tensor([0], dtype=torch.long, device=self.device)

    def load_env(self, colors=None, init_cube_state=None):
        self.env.reset_idx(self.env_id, colors, init_cube_state)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # Resets randomly.
        self.env.reset_idx(self.env_id)

    def reset_to(self, state):
        # Not used in this environment.
        pass

    def render(self, height=None, width=None, camera_name=None, on_screen=False):
        obs_dict = self.env.get_observations()
        return obs_dict["obs"]["images"][0].cpu().numpy()

    def get_observation(self):
        obs_dict = self.env.get_observations()
        return obs_dict

    def is_success(self):
        return False

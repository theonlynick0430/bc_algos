try:
    import omegaconf
    import isaacgymenvs
except ImportError:
    pass
from bc_algos.envs.env_base import BaseEnv
import bc_algos.utils.constants as Const
import torch
import numpy as np


class IsaacGymEnvSimple(BaseEnv):
    """
    Class for interacting with Isaac Gym environment.
    """
    def __init__(
        self,
        env_name,
        obs_key_to_modality,
        render=False,
        use_image_obs=False,
        use_depth_obs=False,
        config=None,
    ):
        """
        Args:
            env_name (str): name of environment

            obs_key_to_modality (dict): dictionary from observation key to modality

            render (bool): if True, environment supports on-screen rendering

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            use_depth_obs (bool): if True, environment is expected to render depth image observations
                on every env.step call. Set this to False for efficiency reasons, if depth
                observations are not required.

            config (dict): parameters for Isaac Gym simulator
        """
        super().__init__(env_name, obs_key_to_modality, render, use_image_obs, use_depth_obs)

        self.config = omegaconf.OmegaConf.create(config)

        self.env = isaacgymenvs.make(
            self.config.seed,
            self.config.task_name,
            1,
            self.config.sim_device,
            self.config.rl_device,
            self.config.graphics_device_id,
            self.config.headless,
            self.config.multi_gpu,
            self.config.capture_video,
            self.config.force_render,
            self.config,
        )

        self.init_cycles = 10
        self.device = self.env.device
        self.env_id = torch.tensor([0], dtype=torch.long, device=self.device)

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
        img = img[..., :-1]
        img = np.moveaxis(img.astype(float), -1, -3)
        img /= 255.
        return img.clip(0., 1.)

    def get_observation(self, di=None, preprocess=True):
        """
        Args:
            di (dict): (optional) current raw observation dictionary from Isaac Gym to wrap and provide 
                as a dictionary. If not provided, will be queried from Isaac Gym.

            preprocess (bool): if True, preprocess observation data

        Returns: observation dictionary from environment. 
        """
        if di is None:
            di = self.env.get_observations()
        di = di["obs"]
        obs = {}
        for k in di:
            if preprocess and (k in self.obs_key_to_modality) and self.obs_key_to_modality[k] == Const.Modality.RGB:
                obs[k] = IsaacGymEnvSimple.preprocess_img(di[k].cpu().numpy())
            elif k in self.obs_key_to_modality:
                obs[k] = di[k].cpu().numpy()
        return obs
    
    def load_env(self, xml):
        """
        Load environment from XML string.

        Args:
            xml (str): scene xml
        """
        return NotImplementedError

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns: observation dictionary after executing action.
        """
        action = torch.tensor(action, device=self.device).float().unsqueeze(0)
        obs = self.env.step(action)
        return self.get_observation(obs)

    def reset(self):
        """
        Reset environment.

        Returns: observation dictionary after resetting environment.
        """
        # reset randomly
        self.env.reset_idx(self.env_id)

        # "warm up" the environment
        for _ in range(self.init_cycles):
            self.step(np.zeros(7))
        
        return self.get_observation()

    def reset_to(self, state):
        """
        Reset to a specific simulator state. For Isaac Gym, state is metadata dictionary.

        Args:
            state (array): current simulator state
        
        Returns: observation dictionary after setting the simulator state.
        """
        assert type(state) == dict, "state must be a dictionary"

        block_colors = state["block_colors"]
        block_init_pose = state["block_init_pose"]
        q_init = torch.from_numpy(state["start_q"]).to(self.device).float().unsqueeze(0)
        block_init_pose = torch.from_numpy(block_init_pose).to(self.device).float().unsqueeze(0)
        self.env.reset_idx(self.env_id, block_colors, block_init_pose)

        # "warm up" the environment
        for _ in range(self.init_cycles):
            self.env.set_arm_dof(self.env_id, q_init)
            obs = self.step(np.zeros(7))
        
        return obs

    def render(self, height=None, width=None, camera_name=None, on_screen=False):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            height (int): height of image to render - only used if mode is "rgb_array"

            width (int): width of image to render - only used if mode is "rgb_array"

            camera_name (str): camera name to use for rendering

            on_screen (bool): if True, render to an on-screen window. otherwise, render
                off-screen to RGB array.

        Returns: rendered image (np.array).
        """
        obs = self.get_observation(preprocess=False)
        return obs["agentview_image"][0]

    def is_success(self):
        """
        Returns: whether the task conditions are reached.
        """
        return False

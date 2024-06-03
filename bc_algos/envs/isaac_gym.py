try:
    import isaacgymenvs
except ImportError:
    pass
from bc_algos.envs.env_base import BaseEnv
import bc_algos.utils.constants as Const
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d
import torch
import numpy as np


BLOCK_RADIUS = 0.025
BLOCK_DIAMETER = BLOCK_RADIUS*2
PICK_THRESH = BLOCK_DIAMETER
PLANAR_THRESH_NEAR = BLOCK_DIAMETER*3
PLANAR_THRESH_STACK = BLOCK_DIAMETER
VERTICAL_THRESH = BLOCK_RADIUS

class IsaacGymEnv(BaseEnv):
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
        use_ortho6D=False,
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

            use_ortho6D (bool): if True, environment uses ortho6D representation for orientation
                
            config (OmegaConf): parameters for Isaac Gym simulator
        """
        super(IsaacGymEnv, self).__init__(
            env_name=env_name,
            obs_key_to_modality=obs_key_to_modality,
            render=render,
            use_image_obs=use_image_obs,
            use_depth_obs=use_depth_obs,
            use_ortho6D=use_ortho6D,
        )

        self.config = config
        self.task = config["task"]["name"]
        self.control_type = config["task"]["env"]["controlType"]

        self.env = isaacgymenvs.make(
            config.seed,
            config.task_name,
            1,
            config.sim_device,
            config.rl_device,
            config.graphics_device_id,
            config.headless,
            config.multi_gpu,
            config.capture_video,
            config.force_render,
            config,
        )

        self.init_cycles = 10
        self.device = self.env.device
        self.env_id = torch.tensor([0], dtype=torch.long).to(self.device)

    @classmethod
    def preprocess_img(cls, img):
        """
        Helper function to preprocess images from Isaac Gym environment.
        Specifically does the following:
        1) Removes alpha (last) channel from images.
        2) Changes shape of images from [H, W, 3] to [3, H, W]
        3) Changes scale of images from [0, 255] to [0, 1]

        Args:
            img: image data of shape [..., H, W, 4]

        Returns: preprocessed @img of shape [..., 3, H, W].
        """
        img = img[..., :-1]
        if isinstance(img, np.ndarray):
            img = np.moveaxis(img, -1, -3).astype(float)
        elif isinstance(img, torch.Tensor):
            img = torch.moveaxis(img, -1, -3).float()
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
            if k in self.obs_key_to_modality:
                obs[k] = di[k][0].cpu().numpy()
                if preprocess:
                    if self.obs_key_to_modality[k] == Const.Modality.RGB:
                        obs[k] = IsaacGymEnv.preprocess_img(obs[k])
        # convert orientation to ortho6D
        if self.use_ortho6D:
            state_quat = obs["robot0_eef_quat"]
            state_mat = quaternion_to_matrix(state_quat)
            state_ortho6D = matrix_to_rotation_6d(state_mat)
            obs["robot0_eef_ortho6D"] = state_ortho6D
        # save for success computation 
        obs["cubes_pos"] = di["cubes_pos"].cpu().numpy()
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
        action = torch.from_numpy(action).to(self.device).float().unsqueeze(0)
        obs = self.env.step(action)
        return self.get_observation(obs)
    
    def warmup(self):
        """
        Warmup environment.

        Returns: observation dictionary after warming up environment.
        """
        if self.control_type == "osc":
            for _ in range(self.init_cycles):
                obs = self.step(np.zeros(7))
        elif self.control_type == "joint_abs":
            obs = self.get_observation()
        return obs

    def reset(self):
        """
        Reset environment.

        Returns: observation dictionary after resetting environment.
        """
        # reset randomly
        self.env.reset_idx(self.env_id)

        return self.warmup()

    def reset_to(self, state):
        """
        Reset to a specific simulator state. For Isaac Gym, state is metadata dictionary.

        Args:
            state (array): current simulator state
        
        Returns: observation dictionary after setting the simulator state.
        """
        assert type(state) == dict, "state must be a dictionary"

        if self.task == "MentalModelsTaskSimple":
            block_colors = state["block_colors"]
            block_init_pose = state["block_init_pose"]
            q_init = torch.from_numpy(state["start_q"]).to(self.device).float().unsqueeze(0)
            block_init_pose = torch.from_numpy(block_init_pose).to(self.device).float().unsqueeze(0)
            self.env.reset_idx(self.env_id, colors=block_colors, init_cube_state=block_init_pose)
        elif self.task == "MentalModelsTask":
            block_indices = torch.from_numpy(state["block_indices"]).to(self.device).long().unsqueeze(0)
            block_radius = torch.from_numpy(state["block_radius"]).to(self.device).float().unsqueeze(0)
            block_type = state["block_types"]
            block_colors = state["block_colors"]
            block_init_pose = state["block_init_pose"]
            block_init_pose[:, 2] += 0.01
            q_init = torch.from_numpy(state["start_q"]).to(self.device).float().unsqueeze(0)
            block_init_pose = torch.from_numpy(block_init_pose).to(self.device).float().unsqueeze(0)
            self.stack = state["offset_index"] == 4
            self.env.reset_idx(self.env_id, active_cube_indices=block_indices, active_cube_radius=block_radius,
                               active_cube_asset_types=block_type, colors=block_colors, init_cube_state=block_init_pose)
        else:
            raise Exception(f"Task {self.task} not supported")
        
        obs = self.warmup()

        # for sucess metrics
        self.src_cube_init_pos = obs["cubes_pos"][0, 0, :]
        self.src_cube_goal_pos = state["cube_pos_goal"]
        self.max_dz = 0.
        self.planar_dist = 0.
        self.vertical_dist = 0.

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
        return obs["agentview_image"]

    def is_success(self):
        """
        Returns: whether the task conditions are reached.
        """
        obs = self.get_observation(preprocess=False)
        src_cube_pos = obs["cubes_pos"][0, 0, :]
        # pick success (max vertical distance achieved)
        self.max_dz = max(self.max_dz, np.abs(src_cube_pos[-1]-self.src_cube_init_pos[-1]))
        pick_success = self.max_dz > PICK_THRESH
        # put success (planar and vertical distance from goal)
        self.planar_dist = np.linalg.norm(self.src_cube_goal_pos[:-1]-src_cube_pos[:-1])
        self.vertical_dist = np.abs(self.src_cube_goal_pos[-1]-src_cube_pos[-1])
        if self.stack:
            put_success = self.planar_dist < PLANAR_THRESH_STACK and self.vertical_dist < VERTICAL_THRESH
        else:
            put_success = self.planar_dist < PLANAR_THRESH_NEAR and self.vertical_dist < VERTICAL_THRESH
        return {
            "pick_success": pick_success, 
            "put_success": put_success,
            "success": pick_success and put_success,
        }
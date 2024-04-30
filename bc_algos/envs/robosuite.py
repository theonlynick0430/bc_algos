"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
""" 
try:
    import robosuite
    import robosuite.utils.transform_utils as T
except ImportError:
    pass
from bc_algos.envs.env_base import BaseEnv
import bc_algos.utils.constants as Const
import json
import numpy as np
import torch
from copy import deepcopy


class RobosuiteEnv(BaseEnv):
    """
    Class for interacting with Robosuite environments (https://github.com/ARISE-Initiative/robosuite).
    """
    def __init__(
        self, 
        env_name,
        obs_key_to_modality,
        render=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        **kwargs,
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

            kwargs (dict): environment specific parameters
        """
        super(RobosuiteEnv, self).__init__(
            env_name=env_name, 
            obs_key_to_modality=obs_key_to_modality,
            render=render,
            use_image_obs=use_image_obs,
            use_depth_obs=use_depth_obs,
        )

        kwargs = deepcopy(kwargs)
        kwargs.update(dict(
            has_renderer=render,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=use_depth_obs,
        ))
        if kwargs["has_offscreen_renderer"]:
            # ensure that we select the correct GPU device for rendering by testing for EGL rendering
            # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
            import egl_probe
            valid_gpu_devices = egl_probe.get_available_devices()
            if len(valid_gpu_devices) > 0:
                kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        self.init_kwargs = deepcopy(kwargs)

        self.env = robosuite.make(self.env_name, **kwargs)

    @classmethod
    def preprocess_img(cls, img):
        """
        Helper function to preprocess images from Robosuite environment. 
        Specifically does the following:
        1) Changes shape of images from [H, W, 3] to [3, H, W]
        2) Changes scale of images from [0, 255] to [0, 1]

        Args: 
            img: image data of shape [..., H, W, 3]

        Returns: preprocessed @img of shape [..., 3, H, W].
        """
        if isinstance(img, np.ndarray):
            img = np.moveaxis(img, -1, -3).astype(float)
        elif isinstance(img, torch.Tensor):
            img = torch.moveaxis(img, -1, -3).float()
        img /= 255.
        return img.clip(0., 1.)
    
    def load_env(self, xml):
        """
        Load environment from XML string.

        Args:
            xml (str): scene xml
        """
        xml = self.env.edit_model_xml(xml)
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns: observation dictionary after executing action.
        """
        obs, _, _, _ = self.env.step(action)
        return self.get_observation(obs)

    def reset(self):
        """
        Reset environment.

        Returns: observation dictionary after resetting environment.
        """
        obs = self.env.reset()
        return self.get_observation(obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (array): current simulator state
        
        Returns: observation dictionary after setting the simulator state.
        """
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        return self.get_observation()
    
    def get_observation(self, di=None):
        """
        Args:
            di (dict): (optional) current raw observation dictionary from Robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from Robosuite.

        Returns: observation dictionary from environment. 
        """
        if di is None:
            di = self.env._get_observations(force_update=True)
        obs = {}
        for k in di:
            if k in self.obs_key_to_modality:
                if self.obs_key_to_modality[k] == Const.Modality.RGB:
                    # by default images from mujoco are flipped in height
                    obs[k] = RobosuiteEnv.preprocess_img(di[k][::-1])
                elif self.obs_key_to_modality[k] == Const.Modality.DEPTH:
                    # by default depth images from mujoco are flipped in height
                    obs[k] = di[k][::-1]
                    if len(obs[k].shape) == 2:
                        obs[k] = obs[k][None, ...] # (1, H, W)
                    assert len(obs[k].shape) == 3 
                    # scale entries in depth map to correspond to real distance.
                    obs[k] = self.get_real_depth_map(obs[k])
                else:
                    obs[k] = di[k]                
        return obs
    
    def is_success(self):
        """
        Returns: whether the task conditions are reached.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            return succ["task"]
        else:
            return succ

    def render(self, height=256, width=256, camera_name="agentview", on_screen=False):
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
        if on_screen:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        else:
            im = self.env.sim.render(height=height, width=width, camera_name=camera_name)
            if self.use_depth_obs:
                # render() returns a tuple when self.use_depth_obs=True
                return im[0][::-1]
            return im[::-1]

    def get_real_depth_map(self, depth_map):
        """
        Reproduced from https://github.com/ARISE-Initiative/robosuite/blob/c57e282553a4f42378f2635b9a3cbc4afba270fd/robosuite/utils/camera_utils.py#L106
        since older versions of Robosuite do not have this conversion from normalized depth values returned by MuJoCo
        to real depth values.
        """
        # Make sure that depth values are normalized
        assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_map * (1.0 - near / far))

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        """
        Obtains camera intrinsic matrix.

        Args:
            camera_name (str): name of camera
            
            camera_height (int): height of camera images in pixels

            camera_width (int): width of camera images in pixels

        Returns:
            K (np.array): 3x3 camera matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
        return K

    def get_camera_extrinsic_matrix(self, camera_name):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the
        viewpoint.
        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html.

        Args:
            camera_name (str): name of camera

        Returns:
            R (np.array): 4x4 camera extrinsic matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        """
        Camera transform matrix to project from world coordinates to pixel coordinates.

        Args:
            camera_name (str): name of camera

            camera_height (int): height of camera images in pixels
            
            camera_width (int): width of camera images in pixels
            
        Returns:
            K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
        """
        R = self.get_camera_extrinsic_matrix(camera_name=camera_name)
        K = self.get_camera_intrinsic_matrix(
            camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
        )
        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        return K_exp @ T.pose_inv(R)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.env_name + "\n" + json.dumps(self.init_kwargs, sort_keys=True, indent=4)

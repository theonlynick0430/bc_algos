"""
This file contains the base class for environment wrappers that are used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Abstract class for interacting with simulation environments. Inherit from 
    this class for different simulators.
    """
    def __init__(
        self,
        env_name, 
        obs_key_to_modality,
        render=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        use_ortho6D=False,
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
        """
        self.env_name = env_name
        self.obs_key_to_modality = obs_key_to_modality
        self._render = render
        self.use_image_obs = use_image_obs
        self.use_depth_obs = use_depth_obs
        self.use_ortho6D = use_ortho6D
    
    @abstractmethod
    def load_env(self, xml):
        """
        Load environment from XML string.

        Args:
            xml (str): scene xml
        """
        return NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns: observation dictionary after executing action.
        """
        return NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset environment.

        Returns: observation dictionary after resetting environment.
        """
        return NotImplementedError

    @abstractmethod
    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (array): current simulator state
        
        Returns: observation dictionary after setting the simulator state.
        """
        return NotImplementedError

    @abstractmethod
    def render(self, height=None, width=None, camera_name=None, on_screen=False):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            height (int): height of image to render - only used if mode is "rgb_array"

            width (int): width of image to render - only used if mode is "rgb_array"

            camera_name (str): camera name to use for rendering

            on_screen (bool): if True, render to an on-screen window. Otherwise, render
                off-screen to RGB array.
        
        Returns: rendered image (np.array).
        """
        return NotImplementedError

    @abstractmethod
    def get_observation(self):
        """
        Returns: observation dictionary from environment. 
        """
        return NotImplementedError
    
    @abstractmethod
    def is_success(self):
        """
        Returns: whether the task conditions are reached.
        """
        return NotImplementedError
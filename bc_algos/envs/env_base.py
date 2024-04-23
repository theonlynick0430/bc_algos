"""
This file contains the base class for environment wrappers that are used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import abc
from enum import StrEnum


class EnvType(StrEnum):
    ROBOSUITE = "robosuite"


class EnvBase(abc.ABC):
    """A base class method for environments used by this repo."""

    @abc.abstractmethod
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
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

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
        self.env_name = env_name
        self.obs_key_to_modality = obs_key_to_modality
        self._render = render
        self.use_image_obs = use_image_obs
        self.use_depth_obs = use_depth_obs

    @abc.abstractmethod
    def load_env(self, xml):
        """
        Load environment from XML string.

        Args:
            xml (str): scene xml
        """
        return NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns: observation dictionary after executing action.
        """
        return NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Reset environment.

        Returns: observation dictionary after resetting environment.
        """
        return NotImplementedError

    @abc.abstractmethod
    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (array): current simulator state
        
        Returns: observation dictionary after setting the simulator state.
        """
        return NotImplementedError

    @abc.abstractmethod
    def render(self, height=None, width=None, camera_name=None, on_screen=False):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            height (int): height of image to render - only used if mode is "rgb_array"

            width (int): width of image to render - only used if mode is "rgb_array"

            camera_name (str): camera name to use for rendering

            on_screen (bool): if True, render to an on-screen window. Otherwise, render
                off-screen to RGB array.
        """
        return NotImplementedError

    @abc.abstractmethod
    def get_observation(self):
        """Get environment observation"""
        return

    @abc.abstractmethod
    def is_success(self):
        """
        Returns: whether the task conditions are reached.
        """
        return

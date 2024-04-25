from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.isaac_gym import IsaacGymDataset
from bc_algos.envs.isaac_gym_simple import IsaacGymEnvSimple
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
import json


class IsaacGymSimpleRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in Isaac Gym environments. 
    """
    def __init__(
            self,
            validset,  
            obs_group_to_key,
            obs_key_to_modality,
            env_cfg_path,
            frame_stack=0,
            closed_loop=True,
            gc=False,
            normalization_stats=None,
            render_video=False,
        ):
        """
        Args:
            validset (SequenceDataset): validation dataset for rollout

            obs_group_to_key (dict): dictionary from observation group to observation key

            obs_key_to_modality (dict): dictionary from observation key to modality

            env_cfg_path (str): path to the config for Isaac Gym simulator

            frame_stack (int): number of stacked frames to be provided as input to policy

            closed_loop (bool): if True, query policy at every timstep and execute first action.
                Otherwise, execute full action chunk before querying the policy again.

            gc (bool): if True, policy uses goals

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            render_video (bool): if True, render rollout on screen
        """
        assert isinstance(validset, IsaacGymDataset)

        self.config = json.load(open(env_cfg_path, "r"))

        super(IsaacGymSimpleRolloutEnv, self).__init__(
            validset=validset,
            obs_group_to_key=obs_group_to_key,
            obs_key_to_modality=obs_key_to_modality,
            frame_stack=frame_stack,
            closed_loop=closed_loop,
            gc=gc,
            normalization_stats=normalization_stats,
            render_video=render_video,
        )

    @classmethod
    def factory(cls, config, validset, normalization_stats=None):
        """
        Create a IsaacGymSimpleRolloutEnv instance from config.

        Args:
            config (addict): config object

            validset (SequenceDataset): validation dataset for rollout

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

        Returns: RolloutEnv instance.
        """
        return cls(
            validset=validset,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            env_cfg_path=config.rollout.env_cfg_path,
            frame_stack=config.dataset.frame_stack,
            closed_loop=config.rollout.closed_loop,
            gc=(config.dataset.goal_mode is not None),
            normalization_stats=normalization_stats,
            render_video=False,
        )

    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time.

        Args:
            demo_id (int): demo id, ie. 0

            t (int): timestep in trajectory

        Returns: goal sequence (np.array) of shape [B=1, T_goal, ...].
        """
        demo_length = self.validset.demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length - 1
        index = self.validset.index_from_timestep(demo_id=demo_id, t=t)
        goal = self.validset[index]["goal"]
        goal = TensorUtils.to_batch(x=goal)
        return goal
    
    def create_env(self):
        """
        Create and return Isaac Gym environment.
        """
        return IsaacGymEnvSimple(
            "MentalModelsTaskSimple",
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            config=self.config,
        )

    def init_demo(self, demo_id):
        """
        Initialize environment for demo by loading models
        and setting simulator state. 
2
        Args:
            demo_id (int): demo id, ie. 0

        Returns: dictionary from observation key to data (np.array) obtained
            from environment after initializing demo
        """
        metadata = self.validset.dataset[demo_id]["metadata"]
        return self.env.reset_to(state=metadata)

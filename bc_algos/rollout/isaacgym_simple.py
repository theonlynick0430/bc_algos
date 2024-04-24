import json

from bc_algos.envs.isaacgym_simple import IsaacGymEnvSimple
from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.isaac_gym import IsaacGymDataset
import bc_algos.utils.constants as Const
import bc_algos.utils.tensor_utils as TensorUtils


class IsaacGymSimpleRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in Isaac Gym environments. 
    """
    def __init__(
            self,
            validset,  
            obs_group_to_key,
            obs_key_to_modality,
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

            frame_stack (int): number of stacked frames to be provided as input to policy

            closed_loop (bool): if True, query policy at every timstep and execute first action.
                Otherwise, execute full action chunk before querying the policy again.

            gc (bool): if True, policy uses goals

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            render_video (bool): if True, render rollout on screen
        """
        assert isinstance(validset, IsaacGymDataset)

    def create_env(self):
        """
        Create and return Isaac Gym environment.
        """
        env_cfg_file = "../config/isaac_gym_env.json"
        config = json.load(open(env_cfg_file, "r"))
        return IsaacGymEnvSimple(
            "MentalModelsTaskSimple",
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            config=config,
        )

    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time.

        Args:
            demo_id (str): demo id, ie. "demo_0"

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

    def init_demo(self, demo_id):
        """
        Initialize environment for demo by loading models
        and setting simulator state. 
2
        Args:
            demo_id (str): demo id, ie. "demo_0"

        Returns: dictionary from observation key to data (np.array) obtained
            from environment after initializing demo
        """
        demo_metadata = self.validset.dataset[demo_id]["metadata"]
        self.env.reset_to(state=demo_metadata)

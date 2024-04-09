from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.robomimic import RobomimicDataset
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.constants as Const
from bc_algos.envs.robosuite import EnvRobosuite
import json


class RobomimicRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in Robomimic environments. 
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

            obs_group_to_key (dict): dictionary mapping observation group to observation key

            obs_key_to_modality (dict): dictionary mapping observation key to modality

            frame_stack (int): number of stacked frames to be provided as input to policy

            closed_loop (bool): if True, query policy at every timstep and execute first action.
                Otherwise, execute full action chunk before querying the policy again.

            gc (bool): if True, policy uses goals

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            render_video (bool): if True, render rollout on screen
        """
        assert isinstance(validset, RobomimicDataset)

        super(RobomimicRolloutEnv, self).__init__(
            validset=validset,
            obs_group_to_key=obs_group_to_key,
            obs_key_to_modality=obs_key_to_modality,
            frame_stack=frame_stack,
            closed_loop=closed_loop,
            gc=gc,
            normalization_stats=normalization_stats,
            render_video=render_video,
        )
    
    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time if goal-conditioned.

        Args: 
            demo_id: demo id, ie. "demo_0"

            t (int): timestep in trajectory

        Returns:
            goal seq np.array of shape [B=1, T=validset.n_frame_stack+1, ...]
        """
        demo_length = self.validset.demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length-1
        index = self.validset.index_from_timestep(demo_id=demo_id, t=t)
        goal = self.validset[index]["goal"]
        goal = TensorUtils.slice(x=goal, dim=0, start=0, end=self.n_frame_stack+1)
        goal = TensorUtils.to_batch(x=goal)
        return goal
        
    def create_env(self):
        """
        Create and return Robosuite environment.
        """
        # load env metadata from training file
        env_meta = json.loads(self.validset.hdf5_file["data"].attrs["env_args"])
        return EnvRobosuite(
            env_name=env_meta["env_name"],
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            **env_meta["env_kwargs"]
        )

    def init_demo(self, demo_id):
        """
        Initialize environment for demo by loading models
        and setting simulator state. 

        Args:
            demo_id: demo id, ie. "demo_0"

        Returns: 
            observation (dict): observation dictionary after initializing demo
        """
        xml = self.validset.hdf5_file[f"data/{demo_id}"].attrs["model_file"]
        init_state = self.validset.hdf5_file[f"data/{demo_id}/states"][0]
        self.env.load_env(xml)
        return self.env.reset_to(init_state)

        







from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.robomimic import RobomimicDataset
from bc_algos.envs.robosuite import RobosuiteEnv
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.constants as Const
import json


class RobomimicRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in Robomimic environments. 
    """
    def __init__(
        self,
        validset,  
        policy,
        obs_group_to_key,
        obs_key_to_modality,
        frame_stack=0,
        closed_loop=True,
        gc=False,
        normalization_stats=None,
        render_video=False,
        video_skip=1,
        terminate_on_success=False,
        horizon=None,
        verbose=False,
    ):
        """
        Args:
            validset (SequenceDataset): validation dataset for rollout

            policy (BC): policy used to query actions

            obs_group_to_key (dict): dictionary from observation group to observation key

            obs_key_to_modality (dict): dictionary from observation key to modality

            frame_stack (int): number of stacked frames to be provided as input to policy

            closed_loop (bool): if True, query policy at every timstep and execute first action.
                Otherwise, execute full action chunk before querying the policy again.

            gc (bool): if True, policy uses goals

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            render_video (bool): if True, render rollout on screen

            video_skip (int): how often to write video frames

            terminate_on_success (bool): if True, terminate episodes early when success is encountered

            horizon (int): (optional) horizon of episodes. If None, use demo length.

            verbose (bool): if True, log rollout stats and visualize error
        """
        assert isinstance(validset, RobomimicDataset)

        super(RobomimicRolloutEnv, self).__init__(
            validset=validset,
            policy=policy,
            obs_group_to_key=obs_group_to_key,
            obs_key_to_modality=obs_key_to_modality,
            frame_stack=frame_stack,
            closed_loop=closed_loop,
            gc=gc,
            normalization_stats=normalization_stats,
            render_video=render_video,
            video_skip=video_skip,
            terminate_on_success=terminate_on_success,
            horizon=horizon,
            verbose=verbose,
        )
    
    def fetch_goal(self, demo_id, t):
        """
        Get goal for timestep @t in demo with @demo_id.

        Args: 
            demo_id (str): demo id, ie. "demo_0"

            t (int): timestep in trajectory

        Returns: goal sequence (np.array) of shape [B=1, T_goal, ...].
        """
        demo_length = self.validset.demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length-1
        index = self.validset.index_from_timestep(demo_id=demo_id, t=t)
        goal = self.validset[index]["goal"]
        goal = TensorUtils.to_batch(x=goal)
        return goal
        
    def create_env(self):
        """
        Create and return Robosuite environment.
        """
        # load env metadata from training file
        env_meta = json.loads(self.validset.hdf5_file["data"].attrs["env_args"])
        return RobosuiteEnv(
            env_name=env_meta["env_name"],
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            **env_meta["env_kwargs"]
        )

    def init_demo(self, demo_id):
        """
        Initialize environment for demo with @demo_id 
        by loading models and setting simulator state. 

        Args:
            demo_id (str): demo id, ie. "demo_0"

        Returns: dictionary from observation key to data (np.array) obtained
            from environment after initializing demo
        """
        xml = self.validset.hdf5_file[f"data/{demo_id}"].attrs["model_file"]
        init_state = self.validset.hdf5_file[f"data/{demo_id}/states"][0]
        self.env.load_env(xml)
        return self.env.reset_to(init_state)

        







from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.isaac_gym import IsaacGymDataset
from bc_algos.envs.isaac_gym import IsaacGymEnv
from bc_algos.utils.misc import load_gzip_pickle
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
import omegaconf
import numpy as np
import json


class IsaacGymRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in Isaac Gym environments. 
    """
    def __init__(
        self,
        validset,  
        policy,
        obs_group_to_key,
        obs_key_to_modality,
        env_cfg_path,
        history=0,
        use_ortho6D=False,
        use_world=False,
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

            history (int): number of frames to be provided as input to policy as history
            
            use_ortho6D (bool): if True, environment uses ortho6D representation for orientation

            use_world (bool): if True, environment represents actions in world frame

            env_cfg_path (str): path to the config for Isaac Gym simulator

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
        assert isinstance(validset, IsaacGymDataset)

        self.config = omegaconf.OmegaConf.create(json.load(open(env_cfg_path, "r")))

        super(IsaacGymRolloutEnv, self).__init__(
            validset=validset,
            policy=policy,
            obs_group_to_key=obs_group_to_key,
            obs_key_to_modality=obs_key_to_modality,
            history=history,
            use_ortho6D=use_ortho6D,
            use_world=use_world,
            closed_loop=closed_loop,
            gc=gc,
            normalization_stats=normalization_stats,
            render_video=render_video,
            video_skip=video_skip,
            terminate_on_success=terminate_on_success,
            horizon=horizon,
            verbose=verbose,
        )
    
    @classmethod
    def factory(cls, config, validset, policy, normalization_stats=None):
        """
        Create a IsaacGymSimpleRolloutEnv instance from config.

        Args:
            config (addict): config object

            validset (SequenceDataset): validation dataset for rollout

            policy (BC): policy used to query actions

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

        Returns: IsaacGymSimpleRolloutEnv instance.
        """
        return cls(
            validset=validset,
            policy=policy,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            env_cfg_path=config.rollout.env_cfg_path,
            history=config.dataset.history,
            use_ortho6D=config.rollout.ortho6D,
            use_world=config.rollout.world,
            closed_loop=config.rollout.closed_loop,
            gc=(config.dataset.goal_mode is not None),
            normalization_stats=normalization_stats,
            render_video=False,
            video_skip=config.rollout.video_skip,
            terminate_on_success=config.rollout.terminate_on_success,
            horizon=config.rollout.horizon,
            verbose=config.rollout.verbose,
        )
    
    def create_env(self):
        """
        Create and return Isaac Gym environment.
        """
        return IsaacGymEnv(
            self.config.task.name,
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            use_ortho6D=self.use_ortho6D,
            config=self.config,
        )

    def init_demo(self, demo_id):
        """
        Initialize environment for demo with @demo_id 
        by loading models and setting simulator state. 

        Args:
            demo_id (int): demo id, ie. 0

        Returns: dictionary from observation key to data (np.array) obtained
            from environment after initializing demo
        """
        run = load_gzip_pickle(filename=self.validset.demo_id_to_run_path(demo_id=demo_id))
        metadata = run["metadata"]
        cubes_pos = run["obs"]["cubes_pos"][0]
        cubes_quat = run["obs"]["cubes_quat"][0]
        cubes_pose = np.concatenate([cubes_pos, cubes_quat], axis=-1)
        metadata.update({
            "block_init_pose": cubes_pose,
            "start_q": run["obs"]["q"][0],
        })
        return self.env.reset_to(state=metadata)

from bc_algos.envs.isaacgym_simple import IsaacGymEnvSimple
from bc_algos.rollout.rollout_env import RolloutEnv
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const


class IsaacGymSimpleRolloutEnv(RolloutEnv):

    def __init__(self,
                 cfg,
                 validset,
                 obs_group_to_key,
                 obs_key_to_modality,
                 frame_stack=0,
                 closed_loop=True,
                 gc=False,
                 normalization_stats=None,
                 render_video=False
                 ):
        self.cfg = cfg
        super().__init__(
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
        Create a RolloutEnv instance from config.

        Args:
            config (addict): config object

            validset (SequenceDataset): validation dataset for rollout

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to
                normalization stats from training dataset

        Returns: RolloutEnv instance.
        """
        return cls(
            config,
            validset=validset,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            frame_stack=config.dataset.frame_stack,
            closed_loop=config.rollout.closed_loop,
            gc=(config.dataset.goal_mode is not None),
            normalization_stats=normalization_stats,
            render_video=False,
        )

    def create_env(self):
        return IsaacGymEnvSimple(
            "MentalModelsTaskSimple",
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            cfg=self.cfg,
        )

    def fetch_goal(self, demo_id, t):
        raise NotImplementedError

    def init_demo(self, demo_id):
        demo_metadata = self.validset.dataset[demo_id]["metadata"]
        self.env.reset_to(state=demo_metadata)

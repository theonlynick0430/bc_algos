import json

from bc_algos.envs.isaacgym_simple import IsaacGymEnvSimple
from bc_algos.rollout.rollout_env import RolloutEnv
import bc_algos.utils.obs_utils as ObsUtils
import bc_algos.utils.constants as Const
from isaacgymenvs.tasks.amp.poselib.poselib import TensorUtils


class IsaacGymSimpleRolloutEnv(RolloutEnv):

    def __init__(self,
                 validset,
                 obs_group_to_key,
                 obs_key_to_modality,
                 frame_stack=0,
                 closed_loop=True,
                 gc=False,
                 normalization_stats=None,
                 render_video=False
                 ):
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

    def create_env(self):
        self.cfg = json.load(open("/home/markvdm/Documents/IsaacGym/bc_algos/config/isaac_gym_env.json", "r"))
        return IsaacGymEnvSimple(
            "MentalModelsTaskSimple",
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(Const.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(Const.Modality.DEPTH in self.obs_key_to_modality.values()),
            cfg=self.cfg,
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
        demo_metadata = self.validset.dataset[demo_id]["metadata"]
        self.env.reset_to(state=demo_metadata)

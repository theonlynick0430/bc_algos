from bc_algos.rollout.rollout_env import RolloutEnv
from bc_algos.dataset.robomimic import RobomimicDataset
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
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
            gc=False,
            render_video=False
            ):
        """
        Args: 
            validset (RobomimicDataset): validation dataset for rollout

            obs_group_to_key (dict): dictionary mapping observation group to observation key

            obs_key_to_modality (dict): dictionary mapping observation key to modality

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 0 (single frame).

            gc (bool): whether or not to condition on goals

            render_video (bool): whether to render rollout on screen
        """
        super(RobomimicRolloutEnv, self).__init__(
            obs_group_to_key=obs_group_to_key,
            obs_key_to_modality=obs_key_to_modality,
            frame_stack=frame_stack,
            gc=gc,
            render_video=render_video,
        )

        assert isinstance(validset, RobomimicDataset)

        self.validset = validset
    
    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time if goal-conditioned.

        Args: 
            demo_id (str): id of the demo, e.g., demo_0

            t (int): timestep in trajectory

        Returns:
            goal seq np.array of shape [B=1, T=validset.n_frame_stack+1, D]
        """
        demo_length = self.validset.get_demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length-1
        index = self.validset.demo_id_to_start_index[demo_id] + t
        goal = self.validset[index]["goal"]
        goal = TensorUtils.slice(x=goal, dim=0, start=0, end=self.n_frame_stack+1)
        goal = TensorUtils.to_batch(x=goal)
        return goal
        
    def create_env(self):
        """
        Create Robosuite environment.
        """
        # load env metadata from training file
        self.env_meta = json.loads(self.validset.hdf5_file["data"].attrs["env_args"])
        self.env = EnvRobosuite(
            env_name=self.env_meta["env_name"],
            obs_key_to_modality=self.obs_key_to_modality,
            render=self.render_video,
            use_image_obs=(ObsUtils.Modality.RGB in self.obs_key_to_modality.values()),
            use_depth_obs=(ObsUtils.Modality.DEPTH in self.obs_key_to_modality.values()),
            **self.env_meta["env_kwargs"]
        )

    def run_rollout(
            self, 
            policy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            horizon=None,
            terminate_on_success=False
        ):
        """
        Run rollout on a single demo and save stats (and video if necessary).

        Args:
            policy (BC instance): policy to use for rollouts

            demo_id (str): id of demo to rollout

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            horizon (int): horizon of rollout episode. If None, use demo length instead.

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        Returns:
            results (dict): dictionary of results with the keys "horizon" and "success"
        """
        super(RobomimicRolloutEnv, self).run_rollout(
            policy=policy,
            demo_id=demo_id,
            video_writer=video_writer, 
            video_skip=video_skip, 
            horizon=horizon,
            terminate_on_success=terminate_on_success
            )
        
        demo_len = self.validset.get_demo_len(demo_id=demo_id)
        horizon = demo_len if horizon is None else horizon
        demo_index = self.validset.demo_id_to_start_index[demo_id]
        
        # switch to eval mode
        policy.eval()

        # init demo
        xml = self.validset.hdf5_file[f"data/{demo_id}"].attrs["model_file"]
        init_state = self.validset.hdf5_file[f"data/{demo_id}/states"][0]
        self.env.load_env(xml)
        obs = self.env.reset_to(init_state)

        # policy inputs from initial observation
        inputs = self.inputs_from_initial_obs(obs=obs, demo_id=demo_id)

        results = {}
        video_count = 0  # video frame counter
        success = False

        for step_i in range(horizon):
            # compute new inputs
            inputs = self.inputs_from_new_obs(x=inputs, obs=obs, demo_id=demo_id, t=demo_index+step_i)

            # get action from policy
            ac = policy(**inputs)

            # play action
            obs = self.env.step(ac)

            success = self.env.is_success()

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = self.env.render()
                    video_writer.append_data(video_img)
                video_count += 1

            # break if success
            if terminate_on_success and success:
                break
        
        results["horizon"] = step_i + 1
        results["success"] = success
        
        return results


        







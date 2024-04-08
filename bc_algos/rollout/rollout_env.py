from bc_algos.dataset.dataset import SequenceDataset
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
from bc_algos.models.policy_nets import BC
import imageio
import time
import os
from collections import OrderedDict


class RolloutEnv:
    """
    Abstract class used to rollout policies. Inherit from this class for 
    different environments. 
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
            validset (SequenceDataset): validation dataset for rollout

            obs_group_to_key (dict): dictionary mapping observation group to observation key

            obs_key_to_modality (dict): dictionary mapping observation key to modality

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 0 (single frame).

            gc (bool): whether or not to condition on goals

            render_video (bool): whether to render rollout on screen
        """
        assert isinstance(validset, SequenceDataset)
        assert validset.pad_frame_stack and validset.pad_seq_length, "rollout requires padding"

        self.validset = validset
        self.obs_group_to_key = obs_group_to_key
        self.obs_key_to_modality = obs_key_to_modality
        self.n_frame_stack = frame_stack
        self.gc = gc
        self.render_video = render_video

        self.env = self.create_env()

    @classmethod
    def factory(cls, config, validset):
        """
        Create a RolloutEnv instance from config.

        Args:
            config (addict): config object

            validset (SequenceDataset): validation dataset for rollout

        Returns:
            RolloutEnv instance
        """
        return cls(
            validset=validset,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            frame_stack=config.dataset.frame_stack,
            gc=(config.dataset.goal_mode is not None),
            render_video=False,
        )

    def create_env(self):
        """
        Create and return environment associated with dataset.
        """
        return NotImplementedError
    
    def inputs_from_initial_obs(self, obs, demo_id):
        """
        Create inputs for model from initial observation.
        For models with history, this requires padding.

        Args: 
            obs (dict): maps obs_key to data of shape [D]

            demo_id: id of the demo

        Returns:
            x (dict): maps obs_group to obs_key to 
                np.array of shape [B=1, T=n_frame_stack+1, D]
        """
        inputs = OrderedDict()
        inputs["obs"] = OrderedDict()
        for obs_key in self.obs_group_to_key["obs"]:
            assert obs_key in obs, f"could not find observation key {obs_key} in observation from environment"
            inputs["obs"][obs_key] = obs[obs_key]
        # add batch, seq dim
        inputs = TensorUtils.to_batch(inputs)
        inputs = TensorUtils.to_sequence(inputs)
        # repeat along seq dim n_frame_stack+1 times to prepare history
        inputs = TensorUtils.repeat_seq(x=inputs, k=self.n_frame_stack+1)
        if self.gc:
            # fetch initial goal
            inputs["goal"] = self.fetch_goal(demo_id=demo_id, t=0)
        return inputs
    
    def inputs_from_new_obs(self, inputs, obs, demo_id, t):
        """
        Update inputs for model by shifting history and inserting new observation.

        Args: 
            inputs (dict): maps obs_group to obs_key to
              np.array of shape [B=1, T=pad_frame_stack+1, D]

            obs (dict): maps obs_key to data of shape [D]

            demo_id: id of the demo

            t (int): timestep in trajectory

        Returns:
            updated input @inputs
        """
        # update input using new obs
        inputs = TensorUtils.shift_seq(x=inputs, k=-1)
        for obs_key in self.obs_group_to_key["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            # only update last seq index to preserve history
            inputs["obs"][obs_key][:, -1, :] = obs[obs_key]
        if self.gc:
            # fetch new goal
            inputs["goal"] = self.fetch_goal(demo_id=demo_id, t=t)
        return inputs
    
    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time if goal-conditioned.

        Args: 
            demo_id: id of the demo

            t (int): timestep in trajectory

        Returns:
            goal seq np.array of shape [B=1, T=validset.n_frame_stack+1, D]
        """
        return NotImplementedError
    
    def init_demo(self, demo_id):
        """
        Initialize environment for demo by loading models
        and setting simulator state. 

        Args:
            demo_id: id of the demo

        Returns: 
            observation (dict): observation dictionary after initializing demo
        """
        return NotImplementedError

    def run_rollout(
            self, 
            policy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            horizon=None,
            terminate_on_success=False,
            device=None,
        ):
        """
        Run rollout on a single demo and save stats (and video if necessary).

        Args:
            policy (BC instance): policy to use for rollouts

            demo_id: id of the demo to rollout

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            horizon (int): horizon of rollout episode. If None, use demo length instead.

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

            device: (optional) device to send tensors to

        Returns:
            results (dict): dictionary of results with the keys "horizon" and "success"
        """
        assert isinstance(policy, BC)

        demo_len = self.validset.get_demo_len(demo_id=demo_id)
        horizon = demo_len if horizon is None else horizon

        # switch to eval mode
        policy.eval()

        obs = self.init_demo(demo_id=demo_id)

        # policy inputs from initial observation
        inputs = self.inputs_from_initial_obs(obs=obs, demo_id=demo_id)

        results = {}
        video_count = 0  # video frame counter
        success = False

        for step_i in range(horizon):
            # compute new inputs
            inputs = self.inputs_from_new_obs(inputs=inputs, obs=obs, demo_id=demo_id, t=step_i)
            x = BC.prepare_inputs(inputs=inputs, device=device)

            # get action from policy
            y = policy(x)
            action = y[0, 0, :].detach().cpu().numpy()

            # play action
            obs = self.env.step(action)

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

    def rollout_with_stats(
            self, 
            policy, 
            demo_id,
            video_dir=None,
            video_writer=None,
            video_skip=5,
            horizon=None,
            terminate_on_success=False, 
            verbose=False,
            device=None,
        ):        
        """
        Configure video writer, run rollout, and log progress. 

        Args:
            policy (RolloutPolicy instance): policy to use for rollouts

            demo_id: id of the demo to rollout

            video_dir (str): (optional) directory to save rollout videos

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            horizon (int): horizon of rollout episode. If None, use demo length instead.

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

            verbose (bool): if True, print results of each rollout

            device: (optional) device to send tensors to

        Returns:
            results (dict): dictionary of results with the keys 
            "time", "horizon", and "success"
        """
        assert isinstance(policy, BC)

        rollout_timestamp = time.time()

        # create video writer
        write_video = video_dir is not None
        video_path = None
        if write_video and video_writer is None:
            video_str = f"{demo_id}.mp4"
            video_path = os.path.join(video_dir, f"{video_str}")
            video_writer = imageio.get_writer(video_path, fps=20)
            print("video writes to " + video_path)
        
        rollout_info = self.run_rollout(
            policy=policy, 
            demo_id=demo_id, 
            video_writer=video_writer, 
            video_skip=video_skip, 
            horizon=horizon,
            terminate_on_success=terminate_on_success, 
            device=device,
        )

        rollout_info["time"] = time.time() - rollout_timestamp
        if verbose:
            horizon = rollout_info["horizon"]
            success = rollout_info["success"]
            print(f"demo={demo_id}, horizon={horizon}, success={success}")

        if write_video:
            video_writer.close()

        return rollout_info

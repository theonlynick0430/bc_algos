from bc_algos.dataset.dataset import SequenceDataset
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
from bc_algos.models.policy_nets import BC
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict
from copy import deepcopy
from abc import abstractmethod


class RolloutEnv:
    """
    Abstract class used to rollout policies. Inherit from this class for 
    different environments. 
    """
    def __init__(
        self,
        validset,  
        policy,
        obs_group_to_key,
        obs_key_to_modality,
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

            history (int): number of frames provided as input to policy as history

            use_ortho6D (bool): if True, environment uses ortho6D representation for orientation

            use_world (bool): if True, environment represents actions in world frame

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
        assert isinstance(validset, SequenceDataset)
        assert isinstance(policy, BC)
        assert validset.pad_history and validset.pad_action_chunk, "rollout requires padding"

        self.validset = validset
        self.policy = policy
        self.obs_group_to_key = obs_group_to_key
        self.obs_key_to_modality = obs_key_to_modality
        self.history = history
        self.use_ortho6D = use_ortho6D
        self.use_world = use_world
        self.closed_loop = closed_loop
        self.gc = gc
        self.normalize = normalization_stats is not None
        self.normalization_stats = normalization_stats
        self.render_video = render_video
        self.video_skip = video_skip
        self.terminate_on_success = terminate_on_success
        self.horizon = horizon
        self.verbose = verbose

        self.env = self.create_env()

    @classmethod
    def factory(cls, config, validset, policy, normalization_stats=None):
        """
        Create a RolloutEnv instance from config.

        Args:
            config (addict): config object

            validset (SequenceDataset): validation dataset for rollout

            policy (BC): policy used to query actions

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

        Returns: RolloutEnv instance.
        """
        return cls(
            validset=validset,
            policy=policy,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
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

    @abstractmethod
    def create_env(self):
        """
        Create and return environment associated with dataset.
        """
        return NotImplementedError
    
    def normalize_obs(self, obs, normalization_stats):
        """
        Normalize @obs according to @normalization_stats.

        Args: 
            obs (dict): dictionary from observation key to data (np.array)

            normalization_stats (dict): nested dictionary from dataset/observation key 
                to a dictionary that contains mean and stdv. 

        Returns: normalized @obs.
        """
        obs = deepcopy(obs)
        for key in normalization_stats:
            if key in obs: obs[key] = ObsUtils.normalize(data=obs[key], normalization_stats=normalization_stats[key])
        return obs
    
    def input_from_initial_obs(self, obs, demo_id, demo):
        """
        Create model input from initial environment observation.
        For models with history, this requires padding.

        Args: 
            obs (dict): dictionary from observation key to data (np.array)

            demo_id: demo id

            demo (dict): nested dictionary returned from @self.validset.load_demo

        Returns: nested dictionary from observation group to observation key
            to data (np.array) of shape [B, T_obs/T_goal, ...]
        """
        if self.normalize:
            obs = self.normalize_obs(obs=obs, normalization_stats=self.normalization_stats)
        
        input = OrderedDict()

        input["obs"] = OrderedDict()
        for obs_key in self.obs_group_to_key["obs"]:
            assert obs_key in obs, f"could not find observation key: {obs_key} in observation from environment"
            input["obs"][obs_key] = obs[obs_key]

        input = TensorUtils.to_batch(x=input)
        input = TensorUtils.to_sequence(x=input)
        input = TensorUtils.repeat_seq(x=input, k=self.history+1) # prepare history

        if self.gc:
            input["goal"] = self.fetch_goal(demo_id=demo_id, demo=demo, t=0)
            
        return input
    
    def input_from_new_obs(self, input, obs, demo_id, demo, t):
        """
        Update model input by shifting history and inserting new observation
        from environment.

        Args: 
            input (dict): nested dictionary from observation group to observation key
                to data (np.array) of shape [B, T_obs/T_goal, ...]

            obs (dict): dictionary from observation key to data (np.array)

            demo_id: demo id

            demo (dict): nested dictionary returned from @self.validset.load_demo

            t (int): timestep in trajectory

        Returns: updated input @input.
        """
        if self.normalize:
            obs = self.normalize_obs(obs=obs, normalization_stats=self.normalization_stats)
        
        input = TensorUtils.shift_seq(x=input, k=-1)

        for obs_key in self.obs_group_to_key["obs"]:
            assert obs_key in obs, f"could not find obs_key: {obs_key} in obs from environment"
            # only update last seq index to preserve history
            input["obs"][obs_key][:, -1, :] = obs[obs_key]
        
        if self.gc:
            input["goal"] = self.fetch_goal(demo_id=demo_id, demo=demo, t=t)

        return input
    
    def fetch_goal(self, demo_id, demo, t):
        """
        Get goal for timestep @t in @demo.

        Args: 
            demo_id (int): demo id, ie. 0

            demo (dict): nested dictionary returned from @self.validset.load_demo

            t (int): timestep in trajectory

        Returns: goal sequence (np.array) of shape [B=1, T_goal, ...].
        """
        demo_length = self.validset.demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length - 1
        goal = self.validset.seq_from_timstep(demo_id=demo_id, demo=demo, t=t)["goal"]
        goal = TensorUtils.to_batch(x=goal)
        return goal
    
    @abstractmethod
    def init_demo(self, demo_id):
        """
        Initialize environment for demo with @demo_id 
        by loading models and setting simulator state. 

        Args:
            demo_id: demo id

        Returns: dictionary from observation key to data (np.array) obtained
            from environment after initializing demo.
        """
        return NotImplementedError
    
    def postprocess_action(self, action, obs):
        """
        Postprocess action if its orientation uses ortho6D or it is represented in the world frame.

        Args: 
            action (tensor): output queried from model of shape [T, action_dim]

            obs (dict): dictionary from observation key to data (np.array)

        Returns: postprocessed @action of shape [T, 7].
        """
        action_pos = action[:, :3]
        action_grip = action[:, -1:]
        if self.use_ortho6D:
            action_ortho6D = action[:, 3:-1]
            action_mat = rotation_6d_to_matrix(action_ortho6D)
        else:
            action_aa = action[:, 3:-1]
            action_mat = axis_angle_to_matrix(action_aa)
        if self.use_world:
            state_pos = torch.from_numpy(obs["robot0_eef_pos"]).to(action.device).unsqueeze(0)
            state_quat = torch.from_numpy(obs["robot0_eef_quat"]).to(action.device).unsqueeze(0)
            state_mat = quaternion_to_matrix(state_quat)
            ee_pose = TensorUtils.se3_matrix(rot=state_mat, pos=state_pos)
            action_pose = TensorUtils.se3_matrix(rot=action_mat, pos=action_pos)
            action_pose = TensorUtils.change_basis(pose=action_pose, transform=ee_pose)
            action_pos = action_pose[:, :3, 3]
            action_mat = action_pose[:, :3, :3]
        action_aa = matrix_to_axis_angle(action_mat)
        action = torch.cat((action_pos, action_aa, action_grip), dim=-1)
        return action

    def run_rollout(self, demo_id, video_writer=None, device=None):
        """
        Run rollout on demo with @demo_id.

        Args:
            demo_id: id of demo to rollout

            video_writer (imageio.Writer): if not None, use video writer object to append frames at 
                rate given by @self.video_skip

            device: (optional) device to send tensors to

        Returns: dictionary of results with the keys "horizon", "success", and "error".
        """
        demo_length = self.validset.demo_len(demo_id=demo_id)
        demo = self.validset.load_demo(demo_id=demo_id)
        horizon = demo_length if self.horizon is None else self.horizon

        # switch to eval mode
        self.policy.eval()

        obs = self.init_demo(demo_id=demo_id)

        # policy input from initial observation
        input = self.input_from_initial_obs(obs=obs, demo_id=demo_id, demo=demo)

        results = {}
        video_count = 0  # video frame counter
        success = False
        error = []

        step_i = 0
        # iterate until horizon reached or termination on success
        while step_i < horizon and not (self.terminate_on_success and success):
            # compute new input
            input = self.input_from_new_obs(input=input, obs=obs, demo_id=demo_id, demo=demo, t=step_i)
            x = BC.prepare_input(input=input, device=device)

            # query policy for actions
            actions = self.policy(x).squeeze(0) # remove batch dim
            pred = actions.detach().cpu().numpy()
            if self.use_ortho6D or self.use_world:
                actions = self.postprocess_action(action=actions, obs=obs)
            actions = actions.detach().cpu().numpy()

            # compute error 
            t = step_i
            if t >= demo_length:
                t = demo_length - 1
            seq = self.validset.seq_from_timstep(demo_id=demo_id, demo=demo, t=t)
            target = seq[self.validset.action_key][self.history:, :]
            e = np.mean(np.abs(target-pred), axis=-1)
            coef = np.full([e.shape[0]], 1.)
            if self.validset.get_pad_mask is True:
                pad_mask = seq["pad_mask"][self.history:]
                coef = pad_mask * coef
            e = np.sum(coef*e)/np.sum(coef)
            error.append(e)

            # slice actions based on planning algorithm
            if self.closed_loop:
                actions = actions[:1, :] # execute only first action
            else:
                actions = actions[:horizon-step_i, :] # execute full action chunk (unless horizon reached)

            # unnormalize actions if necessary
            if self.normalize:
                actions = ObsUtils.unnormalize(
                    data=actions, 
                    normalization_stats=self.normalization_stats[self.validset.action_key],
                )

            # execute actions
            for action in actions:
                obs = self.env.step(action)
                step_i += 1

                metrics = self.env.is_success()
                success = metrics["success"]

                # visualization
                if video_writer is not None:
                    if video_count % self.video_skip == 0:
                        video_img = self.env.render()
                        video_writer.append_data(video_img)
                    video_count += 1

                # break if success
                if self.terminate_on_success and success:
                    break
                
        results["horizon"] = step_i
        results["metrics"] = metrics
        results["error"] = error
        
        return results

    def rollout_with_stats(self, demo_id, video_dir=None, device=None):        
        """
        Run rollout on demo with @demo_id and log progress. 

        Args:
            demo_id: id of demo to rollout

            video_dir (str): (optional) directory to save rollout videos

            device: (optional) device to send tensors to

        Returns: dictionary of results with the keys "duration", "horizon", "success", and "error".
        """
        write_video = video_dir is not None
        video_writer = None
        if write_video:
            video_path = os.path.join(video_dir, f"{demo_id}.mp4")
            video_writer = imageio.get_writer(video_path, fps=24)
            print("video writes to " + video_path)
        
        rollout_timestamp = time.time()
        results = self.run_rollout(
            demo_id=demo_id, 
            video_writer=video_writer, 
            device=device,
        )
        results["duration"] = time.time() - rollout_timestamp

        if write_video:
            video_writer.close()
        
        if self.verbose:
            horizon = results["horizon"]
            metrics = results["metrics"]
            print(f"demo={demo_id}, horizon={horizon}, metrics={metrics}")
            if write_video:
                img_path = os.path.join(video_dir, f"{demo_id}_error.png")
                error = np.array(results["error"])
                t = np.arange(error.shape[0])
                plt.figure()
                plt.plot(t, error)
                plt.title("rollout error")
                plt.xlabel("time")
                plt.ylabel("error")
                plt.savefig(img_path)              

        return results

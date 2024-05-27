from bc_algos.dataset.dataset import SequenceDataset
from bc_algos.utils.misc import load_gzip_pickle
import os


class IsaacGymDataset(SequenceDataset):
    """
    Class for fetching sequences of experience from Isaac Gym dataset.
    Length of the fetched sequence is equal to (@self.frame_stack + @self.seq_length)
    """

    def __init__(
        self,
        path,
        obs_key_to_modality,
        obs_group_to_key,
        action_key,
        history=0,
        action_chunk=1,
        pad_history=True,
        pad_action_chunk=True,
        get_pad_mask=True,
        goal_mode=None,
        num_subgoal=None,
        normalize=False,
        normalization_stats=None,
        filter_by_attribute=None,
        demos=None,
    ):
        """
        Args:
            path (str): path to dataset directory

            obs_key_to_modality (dict): dictionary from observation key to modality

            obs_group_to_key (dict): dictionary from observation group to observation key

            action_key (str): key to dataset actions

            history (int): number of frames to fetch for history

            action_chunk (int): number of frames to fetch for action prediction

            pad_history (bool): if True, pad sequence for history at the beginning of a demo. For example, 
                padding for @history=3 and @action_chunk=1, the first sequence would be (s_0, s_0, s_0, s_0) 
                instead of (s_0, s_1, s_2, s_3). 

            pad_action_chunk (bool): if True, pad sequence for action_chunk at the end of a demo. For example, 
                padding for @history=0 and @action_chunk=3, the last sequence would be (s_T, s_T, s_T) 
                instead of (s_T-2, s_T-1, s_T). 

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (GoalMode): (optional) type of goals to be fetched. 
                If GoalMode.LAST, provide last observation as goal for each frame.
                If GoalMode.SUBGOAL, provide an intermediate observation as goal for each frame.
                If GoalMode.FULL, provide all subgoals for a single batch.
                Defaults to None, or no goals. 

            num_subgoal (int): (optional) number of subgoals for each trajectory.
                Defaults to None, which indicates that every frame in trajectory is also a subgoal. 
                Assumes that @num_subgoal <= min trajectory length.

            normalize (bool): if True, normalize data according to mean and stdv 
                computed from the dataset or provided in @normalization_stats.

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            filter_by_attribute (str): (optional) if provided, use the provided filter key 
                to look up a subset of demos to load

            demos (array): (optional) if provided, only load demos with these selected ids
        """
        self.path = path
        self.filter_by_attribute = filter_by_attribute
        self._demos = demos

        super(IsaacGymDataset, self).__init__(
            obs_key_to_modality=obs_key_to_modality,
            obs_group_to_key=obs_group_to_key,
            action_key=action_key,
            history=history,
            action_chunk=action_chunk,
            pad_history=pad_history,
            pad_action_chunk=pad_action_chunk,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            num_subgoal=num_subgoal,
            normalize=normalize,
            normalization_stats=normalization_stats,
        )

    def demo_id_to_run_path(self, demo_id):
        """
        Args: 
            demo_id (int): demo id, ie. 0
        
        Returns: run path associated with @demo_id.
        """
        return os.path.join(self.path, f"run_{demo_id}.pkl.gzip")

    @property
    def demos(self):
        """
        Returns: all demo ids.
        """
        if self._demos is None:
            if self.filter_by_attribute is not None:
                split_path = os.path.join(self.path, "split.pkl.gzip")
                split = load_gzip_pickle(filename=split_path)
                self._demos = split[self.filter_by_attribute]
            else:
                self._demos = [i for i in range(len(os.listdir(self.path))) if
                               os.path.isfile(self.demo_id_to_run_path(demo_id=i))]
        return self._demos
    
    def load_demo(self, demo_id):
        """
        Load demo with @demo_id into memory.

        Args: 
            demo_id: demo id

        Returns: nested dictionary with the following format:
        {
            dataset_key: data (np.array) of shape [T, ...]
            ...
            obs_key: data (np.array) of shape [T, ...]
            ...
            "length": length of trajectory
        }
        """
        run = load_gzip_pickle(filename=self.demo_id_to_run_path(demo_id=demo_id))
        demo = {obs_key: run["obs"][obs_key] for obs_key in self.obs_keys}
        demo[self.action_key] = run["policy"][self.action_key]
        demo["length"] = run["metadata"]["num_steps"]-1
        return demo

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__) + "(\n"
        msg += super(IsaacGymDataset, self).__repr__()
        msg += "\tfilter_key={}\n" + ")"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        msg = msg.format(filter_key_str)
        return msg

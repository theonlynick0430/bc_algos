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
        dataset_keys,
        frame_stack=0,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        goal_mode=None,
        num_subgoal=None,
        filter_by_attribute=None,
        demos=None,
    ):
        """
        Args:
            path (str): path to dataset directory

            obs_key_to_modality (dict): dictionary from observation key to modality

            obs_group_to_key (dict): dictionary from observation group to observation key

            dataset_keys (array): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): number of stacked frames to fetch. Defaults to 0 (no stacking).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): if True, pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): if True, to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

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
            dataset_keys=dataset_keys,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            num_subgoal=num_subgoal,
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
    
    @property
    def normalization_stats(self):
        """
        Returns: if dataset is normalized, a nested dictionary from dataset/observation key 
            to a dictionary that contains keys "mean" and "stdv". Otherwise, None.
        """
        stats_path = os.path.join(self.path, "normalization_stats.pkl.gzip")
        if os.path.isfile(stats_path):
            return load_gzip_pickle(filename=stats_path)
        else:
            return None
    
    def demo_len(self, demo_id):
        """
        Args: 
            demo_id (int): demo id, ie. 0
        
        Returns: length of demo with @demo_id.
        """
        run = load_gzip_pickle(filename=self.demo_id_to_run_path(demo_id=demo_id))
        return run["metadata"]["num_steps"]-1
    
    def load_demo(self, demo_id):
        """
        Load demo with @demo_id into memory. 

        Args: 
            demo_id (int): demo id, ie. 0
        
        Returns: nested dictionary from dataset/observation key to
            data (np.array) of shape [T, ...]
        """
        run = load_gzip_pickle(filename=self.demo_id_to_run_path(demo_id=demo_id))
        # get observations
        demo = {obs_key: run["obs"][obs_key] for obs_key in self.obs_keys}
        # get other dataset keys
        for dataset_key in self.dataset_keys:
            demo[dataset_key] = run["policy"][dataset_key]
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

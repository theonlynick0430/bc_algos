from bc_algos.dataset.dataset import SequenceDataset
import h5py
from tqdm import tqdm


class RobomimicDataset(SequenceDataset):
    """
    Class for fetching sequences of experience from Robomimic dataset.
    Length of the fetched sequence is equal to (@self.frame_stack + @self.seq_length)
    """
    def __init__(
        self,
        path,
        obs_key_to_modality,
        obs_group_to_key,
        action_key,
        frame_stack=0,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
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
            path (str): path to dataset

            obs_key_to_modality (dict): dictionary from observation key to modality

            obs_group_to_key (dict): dictionary from observation group to observation key

            action_key (str): key to dataset actions

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

            normalize (bool): if True, normalize data according to mean and stdv 
                computed from the dataset or provided in @normalization_stats.

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

            filter_by_attribute (str): (optional) if provided, use the provided filter key 
                to look up a subset of demos to load

            demos (array): (optional) if provided, only load demos with these selected ids
        """
        self.path = path
        self._hdf5_file = None
        self.filter_by_attribute = filter_by_attribute
        self._demos = demos

        super(RobomimicDataset, self).__init__(
            obs_key_to_modality=obs_key_to_modality,
            obs_group_to_key=obs_group_to_key,
            action_key=action_key,
            frame_stack=frame_stack,
            seq_length=seq_length, 
            pad_frame_stack=pad_frame_stack, 
            pad_seq_length=pad_seq_length, 
            get_pad_mask=get_pad_mask, 
            goal_mode=goal_mode, 
            num_subgoal=num_subgoal,
            normalize=normalize,
            normalization_stats=normalization_stats,
        )

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.path, 'r', swmr=True, libver='latest')
        return self._hdf5_file  

    @property
    def demos(self):
        """
        Returns: all demo ids.
        """
        if self._demos is None:
            if self.filter_by_attribute is not None:
                self._demos = [elem.decode("utf-8") 
                               for elem in self.hdf5_file[f"mask/{self.filter_by_attribute}"][:]]
            else:
                self._demos = list(self.hdf5_file["data"].keys())
        return self._demos
    
    def load_dataset(self):
        """
        Load dataset into memory.

        Returns: nested dictionary with the following format:
        {
            demo_id: {
                dataset_key: data (np.array) of shape [T, ...]
                ...
                obs_key: data (np.array) of shape [T, ...]
                ...
                "steps": length of trajectory
            }
            ...
        }
        """
        dataset = {}

        with tqdm(total=self.num_demos, desc="loading dataset into memory", unit='demo') as progress_bar:
            for demo_id in self.demos:
                dataset[demo_id] = {}
                # get observations
                dataset[demo_id] = {obs_key: self.hdf5_file[f"data/{demo_id}/obs/{obs_key}"][()] for obs_key in self.obs_keys}
                # get actions
                dataset[demo_id][self.action_key] = self.hdf5_file[f"data/{demo_id}/{self.action_key}"][()]
                dataset[demo_id]["steps"] = self.hdf5_file[f"data/{demo_id}"].attrs["num_samples"] 

                progress_bar.update(1)

        return dataset 
    
    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__) + "(\n"
        msg += super(RobomimicDataset, self).__repr__()
        msg += "\tfilter_key={}\n"+ ")"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        msg = msg.format(filter_key_str)
        return msg
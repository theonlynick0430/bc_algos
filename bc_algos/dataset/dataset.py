import numpy as np
import torch.utils.data
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
from bc_algos.utils.constants import GoalMode
import bc_algos.utils.constants as Const
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import OrderedDict


class SequenceDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract class for fetching sequences of experience. Inherit from this class for 
    different dataset formats. 
    Length of the fetched sequence is equal to (@self.frame_stack + @self.seq_length)
    """
    def __init__(
        self,
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
        preprocess=False,
        normalize=True,
    ):
        """
        Args:
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
    
            preprocess (bool): if True, preprocess data while loading into memory

            normalize (bool): if True, normalize data using mean and stdv from dataset
        """
        self.obs_key_to_modality = obs_key_to_modality
        self.obs_group_to_key = obs_group_to_key
        self.obs_keys = list(set([obs_key for obs_group in obs_group_to_key.values() for obs_key in obs_group]))
        self.dataset_keys = list(dataset_keys)

        assert frame_stack >= 0
        self.frame_stack = frame_stack
        assert seq_length >= 1
        self.seq_length = seq_length

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        if goal_mode is not None:
            assert "goal" in obs_group_to_key, "observation group: goal must exist to provide goals"
            assert goal_mode in [GoalMode.LAST, GoalMode.SUBGOAL, GoalMode.FULL], f"goal_mode: {goal_mode} not supported"
            if goal_mode == GoalMode.FULL:
                assert num_subgoal is not None, "goal_mode: full requires the number of subgoals to be specified"
        self.goal_mode = goal_mode
        self.num_subgoal = num_subgoal

        self.dataset = self.load_dataset(preprocess=preprocess)

        self.normalize = normalize
        if normalize:
            self._normalization_stats = self.compute_normalization_stats(dataset=self.dataset)
            self.normalize_dataset(dataset=self.dataset, normalization_stats=self._normalization_stats)

        self.load_demo_info()
        self.cache_index()

    @classmethod
    def factory(cls, config, train=True):
        """
        Create a SequenceDataset instance from config.

        Args:
            config (addict): config object

            train (bool): if True, use kwargs for training dataset. 
                Otherwise, use kwargs for validation dataset.

        Returns: SequenceDataset instance.
        """
        kwargs = config.dataset.kwargs.train if train else config.dataset.kwargs.valid
        return cls(
            path=config.dataset.path,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            dataset_keys=config.dataset.dataset_keys,
            frame_stack=config.dataset.frame_stack,
            seq_length=config.dataset.seq_length,
            pad_frame_stack=config.dataset.pad_frame_stack,
            pad_seq_length=config.dataset.pad_seq_length,
            get_pad_mask=config.dataset.get_pad_mask,
            goal_mode=config.dataset.goal_mode,
            num_subgoal=config.dataset.num_subgoal,
            preprocess=config.dataset.preprocess,
            normalize=config.dataset.normalize,
            **kwargs
        )

    @property
    @abstractmethod
    def demos(self):
        """
        Returns: all demo ids.
        """
        return NotImplementedError      

    @property
    def num_demos(self):
        """
        Returns: number of demos.
        """
        return len(self.demos)

    @property
    def gc(self):
        """
        Returns: whether this dataset contains goals.
        """
        return self.goal_mode is not None

    @abstractmethod
    def load_dataset(self, preprocess):
        """
        Load dataset into memory.

        Args: 
            preprocess (bool): if True, preprocess data while loading into memory

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
        return NotImplementedError
    
    def demo_len(self, demo_id):
        """
        Args: 
            demo_id: demo id
        
        Returns: length of demo with @demo_id.
        """
        return self.dataset[demo_id]["steps"]
    
    def compute_normalization_stats(self, dataset):
        """
        Compute the mean and stdv for items in @dataset.

        Args: 
            dataset (dict): dataset returned from @self.load_dataset

        Returns: nested dictionary from dataset/observation key 
            to a dictionary that contains mean and stdv. 
        """
        traj_dict = {}
        merged_stats = {}

        # don't compute normalization stats for RGB data since we use backbone encoders
        # with their own normalization stats
        keys = [obs_key for obs_key in self.obs_keys if self.obs_key_to_modality[obs_key] != Const.Modality.RGB] + self.dataset_keys

        with tqdm(total=self.num_demos, desc="computing normalization stats", unit="demo") as progress_bar:
            for i, demo_id in enumerate(self.demos):
                traj_dict = {key: dataset[demo_id][key] for key in keys}
                if i == 0:
                    merged_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                else:
                    traj_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                    merged_stats = ObsUtils.aggregate_traj_stats(traj_stats_a=merged_stats, traj_stats_b=traj_stats)

                progress_bar.update(1)
        
        return ObsUtils.compute_normalization_stats(traj_stats=merged_stats, tol=1e-3)

    def normalize_dataset(self, dataset, normalization_stats):
        """
        Normalize items in @dataset in place according to @normalization_stats.

        Args: 
            dataset (dict): dataset returned from @self.load_dataset

            normalization_stats (dict): normalization stats returned from @self.compute_normalization_stats
        """
        with tqdm(total=self.num_demos, desc="normalizing data", unit="demo") as progress_bar:
            for demo_id in self.demos:
                for key in normalization_stats:
                    dataset[demo_id][key] = ObsUtils.normalize(data=dataset[demo_id][key], normalization_stats=normalization_stats[key])
                    
                progress_bar.update(1)

    @property
    def normalization_stats(self):
        """
        Returns: if @self.normalize is True, a nested dictionary from dataset/observation key
            to a dictionary that contains mean and stdv. Otherwise, None.
        """
        if self.normalize:
            return self._normalization_stats
        else:
            return None

    def load_demo_info(self):
        """
        Populate internal data structures.
        """
        self.total_num_sequences = 0
        # array from index in total_num_sequences to demo_id
        self.index_to_demo_id = []
        # dictionary from demo_id to start index in total_num_sequences
        self.demo_id_to_start_index = {}
        # dictionary from demo_id to length of demo in data
        self.demo_id_to_demo_length = {}
        
        for demo_id in self.demos:
            demo_length = self.demo_len(demo_id=demo_id)
            self.demo_id_to_start_index[demo_id] = self.total_num_sequences
            self.demo_id_to_demo_length[demo_id] = demo_length

            num_sequences = demo_length
            if not self.pad_frame_stack:
                num_sequences -= self.frame_stack
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            for _ in range(num_sequences):
                self.index_to_demo_id.append(demo_id)
                self.total_num_sequences += 1   

    def index_from_timestep(self, demo_id, t):
        """
        Args: 
            demo_id: demo id

            t (int): timestep in demo

        Returns: get_item index for timestep @t in demo with @demo_id.
        """
        return self.demo_id_to_start_index[demo_id] + t

    def get_data_seq_index(self, demo_id, index_in_demo):
        """
        Get sequence indices and pad mask to extract data from a demo. 

        Args:
            demo_id: demo id

            index_in_demo (int): beginning index of the sequence wrt the demo

        Returns: data sequence indices and pad mask.
        """
        demo_length = self.demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - self.frame_stack)
        seq_end_index = min(demo_length, index_in_demo + self.seq_length)
        seq_index = np.arange(seq_begin_index, seq_end_index)

        # determine sequence padding
        seq_begin_pad = max(0, self.frame_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + self.seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        seq_index = TensorUtils.pad(seq=seq_index, dim=0, padding=(seq_begin_pad, seq_end_pad))
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
            
        return seq_index, pad_mask 
    
    def get_goal_seq_index(self, demo_id, data_seq_index):
        """
        Get sequence indices to extract goals from a demo. 

        Args:
            demo_id: demo id

            data_seq_index (array): sequence indices

        Returns: goal sequence indices.
        """
        demo_length = self.demo_id_to_demo_length[demo_id]

        if self.goal_mode == GoalMode.LAST:
            goal_index = np.array([-1])

        elif self.goal_mode == GoalMode.SUBGOAL:
            if self.num_subgoal is None:
                goal = np.arange(1, demo_length+1)
            else:
                subgoal = np.linspace(0, demo_length, self.num_subgoal+1, dtype=np.uint32)
                goal = np.repeat(subgoal[1:], np.diff(subgoal))
            goal_index = goal[data_seq_index[self.frame_stack:]]
            
        elif self.goal_mode == GoalMode.FULL:
            goal_index = np.linspace(0, demo_length, self.num_subgoal+1, dtype=np.uint32)[1:]

        return goal_index
    
    def cache_index(self):
        """
        Cache all index required for get_item calls to speed up training and reduce memory.
        """
        # index cache for get_item calls
        self.index_cache = []

        with tqdm(total=len(self), desc="caching index", unit='demo') as progress:
            for index in range(len(self)):
                demo_id = self.index_to_demo_id[index]
                offset = 0 if self.pad_frame_stack else self.frame_stack
                demo_index = index - self.demo_id_to_start_index[demo_id] + offset

                data_seq_index, pad_mask = self.get_data_seq_index(demo_id=demo_id, index_in_demo=demo_index)
                item = [
                    data_seq_index, 
                    pad_mask if self.get_pad_mask else None,
                    self.get_goal_seq_index(demo_id=demo_id, data_seq_index=data_seq_index) if self.gc else None
                ]
                self.index_cache.append(item)

                progress.update(1)

    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of dataset items from a demo.

        Args:
            demo_id (str): demo id, ie. "demo_0"

            keys (array): keys to extract

            seq_index (array): sequence indices

        Returns: ordered dictionary from key to extracted data.
        """
        seq = OrderedDict()
        for k in keys:
            data = self.dataset[demo_id][k]
            seq[k] = data[seq_index]
        return seq

    def get_item(self, index):
        """
        Main implementation of getitem.

        Args: 
            index (int): index of dataset item to fetch

        Returns: nested dictionary with three possible items:
        
            1) obs (dict): dictionary from observation key to data (np.array)
                of shape [T = @self.frame_stack + @self.seq_length, ...]
            
            2) goal (dict): dictionary from observation key to data (np.array) of shape [T_goal, ...]

            3) pad_mask (np.array): mask of shape [T = @self.frame_stack + @self.seq_length] 
                indicating which frames are padding
        """
        demo_id = self.index_to_demo_id[index]
        cache = self.index_cache[index]

        data_seq_index, pad_mask, goal_index = cache
        item = self.get_data_seq(demo_id=demo_id, keys=self.dataset_keys, seq_index=data_seq_index)
        item["obs"] = self.get_data_seq(demo_id=demo_id, keys=self.obs_group_to_key["obs"], seq_index=data_seq_index)
        if self.gc:
            item["goal"] = self.get_data_seq(demo_id=demo_id, keys=self.obs_group_to_key["goal"], seq_index=goal_index)
        if self.get_pad_mask:
            item["pad_mask"] = pad_mask

        return item
    
    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences
    
    def __getitem__(self, index):
        """
        Returns: dataset sequence for @index.
        """
        return self.get_item(index)
    
    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = "\tframe_stack={}\n\tseq_length={}\n\tpad_frame_stack={}\n\tpad_seq_length={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n"
        msg += "\tgoal_mode={}\n\tnum_subgoal={}\n"
        msg += "\tnormalize={}\n"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        num_subgoal_str = self.num_subgoal if self.num_subgoal is not None else "none"
        msg = msg.format(
            self.frame_stack, self.seq_length, self.pad_frame_stack, self.pad_seq_length,
            self.num_demos, self.total_num_sequences,
            goal_mode_str, num_subgoal_str,
            self.normalize,
        )
        return msg
"""
This file contains abstract Dataset classes that are used by torch dataloaders
to fetch batches from datasets.
"""
import numpy as np
import torch.utils.data
import bc_algos.utils.tensor_utils as TensorUtils
import bc_algos.utils.obs_utils as ObsUtils
import os
from tqdm import tqdm
from abc import ABC, abstractmethod


class SequenceDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract class for fetching sequences of experience. Inherit from this class for 
    different dataset formats. 
    Length of the fetched sequence is equal to (@frame_stack + @seq_length)
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
        get_pad_mask=False,
        goal_mode=None,
        num_subgoal=None,
        demos=None,
        preprocess=False,
        normalize=False,
    ):
        """
        Args:
            path (str): path to dataset 

            obs_key_to_modality (dict): dictionary mapping observation key to modality

            obs_group_to_key (dict): dictionary from observation group to observation key

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 0 (single frame).

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

            goal_mode (str): either "last", "subgoal", or None. Defaults to None, which is to not fetch goals

            num_subgoal (int): Required if goal_mode is "subgoal". Number of subgoals provided for each trajectory.
                Defaults to None, which indicates that every state is also a subgoal. Assume num_subgoal <= min length of traj.
    
            demos (list): if provided, use only load these selected demos

            preprocess (bool): if True, preprocess data while loading it into memory

            normalize (bool): if True, normalize data using mean and stdv from dataset
        """
        self.path = os.path.expanduser(path)
        self.obs_key_to_modality = obs_key_to_modality
        self.obs_group_to_key = obs_group_to_key
        self.obs_keys = list(set([key for keys in self.obs_group_to_key.values() for key in keys]))
        self.dataset_keys = list(dataset_keys)
        self._demos = demos

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 0
        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.goal_mode = goal_mode
        self.num_subgoal = num_subgoal
        if self.goal_mode is not None:
            assert self.goal_mode in ["last", "subgoal"]

        # INTERNAL DATA STRUCTURES
            
        self.total_num_sequences = 0
        # maps index in total_num_sequences to demo_id
        self.index_to_demo_id = []
        # maps demo_id to start index in total_num_sequences
        self.demo_id_to_start_index = dict()
        # maps demo_id to length of demo in data
        self.demo_id_to_demo_length = dict()
        # index cache for get_item calls
        self.index_cache = []
        # data loaded from dataset
        self.dataset = None
        # maps dataset/observation key to normalization stats
        self.normalization_stats = None

        self.load_demo_info()
        self.cache_index()

        self.load_dataset_in_memory(preprocess=preprocess)

        self.normalize = normalize
        if normalize:
            self.compute_normalization_stats()
            self.normalize_data()

    @classmethod
    def factory(cls, config, train=True):
        """
        Create a SequenceDataset instance from config.

        Args:
            config (addict): config object

            train (bool): if True, use kwargs for training dataset. 
                Otherwise, use kwargs for validation dataset.

        Returns:
            SequenceDataset instance
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
        return "goal" in self.obs_group_to_key and self.goal_mode in ["last", "subgoal"]

    @abstractmethod
    def demo_len(self, demo_id):
        """
        Args: 
            demo_id: demo id, ie. "demo_0"
        
        Returns: length of demo with @demo_id.
        """
        return NotImplementedError
    
    def index_from_timestep(self, demo_id, t):
        """
        Args: 
            demo_id: demo id, ie. "demo_0"

            t (int): timestep in demo

        Returns: get_item index for timestep @t in demo with @demo_id.
        """
        return self.demo_id_to_start_index[demo_id] + t

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
        msg = "\tpath={}\n"
        msg += "\tframe_stack={}\n\tseq_length={}\n\tpad_frame_stack={}\n\tpad_seq_length={}\n"
        msg += "\tgoal_mode={}\n\tnum_subgoal={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        num_subgoal_str = self.num_subgoal if self.num_subgoal is not None else "none"
        msg = msg.format(
            self.path,
            self.n_frame_stack, self.seq_length, self.pad_frame_stack, self.pad_seq_length, 
            goal_mode_str, num_subgoal_str,
            self.num_demos, self.total_num_sequences
        )
        return msg
    
    def load_demo_info(self):
        """
        Populate internal data structures.
        """
        for demo_id in self.demos:
            demo_length = self.demo_len(demo_id=demo_id)
            self.demo_id_to_start_index[demo_id] = self.total_num_sequences
            self.demo_id_to_demo_length[demo_id] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= self.n_frame_stack
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack + self.seq_length)

            for _ in range(num_sequences):
                self.index_to_demo_id.append(demo_id)
                self.total_num_sequences += 1   

    @abstractmethod
    def load_dataset_in_memory(self, preprocess):
        """
        Load the dataset into memory.

        Args: 
            preprocess (bool): if True, preprocess data while loading it into memory
        """
        return NotImplementedError
    
    @abstractmethod
    def compute_normalization_stats(self):
        """
        Compute the mean and stdv for dataset items and store stats at @self.normalization_stats.
        The format for @self.normalization_stats should be a dictionary that maps from dataset/observation
        key to a dictionary that contains mean and stdv. 

        Example:
        {
            "actions": {
                "mean": ...,
                "stdv": ...
            }
        }
        """
        return NotImplementedError
    
    @abstractmethod
    def normalize_data(self):
        """
        Normalize dataset items according to @self.normalization_stats.
        """
        return NotImplementedError
    
    def get_item(self, index):
        """
        Main implementation of getitem.
        """
        demo_id = self.index_to_demo_id[index]
        cache = self.index_cache[index]
        data_seq_index = cache[0]
        meta = self.get_data_seq(demo_id=demo_id, keys=self.dataset_keys, seq_index=data_seq_index)
        meta["obs"] = self.get_data_seq(demo_id=demo_id, keys=self.obs_group_to_key["obs"], seq_index=data_seq_index)
        if self.gc:
            goal_index = cache[1]
            meta["goal"] = self.get_data_seq(demo_id=demo_id, keys=self.obs_group_to_key["goal"], seq_index=goal_index)
        if self.get_pad_mask:
            pad_mask = cache[2]
            meta["pad_mask"] = pad_mask
        return meta

    def cache_index(self):
        """
        Cache all index required for get_item calls to speed up training and reduce memory.
        """
        self.index_cache = []
        with tqdm(total=len(self), desc="caching index", unit='demo') as progress:
            for index in range(len(self)):
                demo_id = self.index_to_demo_id[index]
                # convert index in total_num_sequences to index in data
                start_offset = 0 if self.pad_frame_stack else self.n_frame_stack
                demo_index = index - self.demo_id_to_start_index[demo_id] + start_offset
                data_seq_index, pad_mask = self.get_data_seq_index(demo_id=demo_id, index_in_demo=demo_index)
                item = [data_seq_index]
                if self.gc:
                    item.append(self.get_goal_seq_index(demo_id=demo_id, data_seq_index=data_seq_index))
                if self.get_pad_mask:
                    item.append(pad_mask)
                self.index_cache.append(item)
                progress.update(1)
    
    @abstractmethod
    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of dataset items from a demo.

        Args:
            demo_id: demo id, ie. "demo_0"

            keys (tuple): keys to extract

            seq_index (list): sequence indices

        Returns: ordered dictionary of extracted items.
        """
        return NotImplementedError

    def get_data_seq_index(self, demo_id, index_in_demo):
        """
        Get sequence indices and pad mask to extract data from a demo. 

        Args:
            demo_id: demo id, ie. "demo_0"

            index_in_demo (int): beginning index of the sequence wrt the demo

        Returns: data sequence indices and pad mask.
        """
        demo_length = self.demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - self.n_frame_stack)
        seq_end_index = min(demo_length, index_in_demo + self.seq_length)
        seq_index = np.arange(seq_begin_index, seq_end_index, dtype=np.int32)

        # determine sequence padding
        seq_begin_pad = max(0, self.n_frame_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + self.seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        seq_index = TensorUtils.pad(seq=seq_index, dim=0, padding=(seq_begin_pad, seq_end_pad))
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)
            
        return seq_index, pad_mask 
    
    def get_goal_seq_index(self, demo_id, data_seq_index):
        """
        Get sequence indices to extract goals from a demo. 

        Args:
            demo_id: demo id, ie. "demo_0"

            data_seq_index (list): sequence indices

        Returns: goal sequence indices.
        """
        demo_length = self.demo_id_to_demo_length[demo_id]
        goal_index = None
        if self.goal_mode == "last":
            goal_index = np.full((demo_length), -1) # every goal is the last state
        elif self.goal_mode == "subgoal" and self.num_subgoal is None:
            goal_index = np.arange(1, demo_length+1) # goal is the next state
        elif self.goal_mode == "subgoal":
            # create evenly spaced subgoals
            subgoal_index = np.linspace(0, demo_length, self.num_subgoal+1, dtype=int)
            repeat = np.diff(subgoal_index)
            goal_index = np.array([index for i, index in enumerate(subgoal_index[1:]) for _ in range(repeat[i])])

        goal_index = goal_index[data_seq_index]

        return goal_index
import bc_algos.utils.obs_utils as ObsUtils
from bc_algos.utils.constants import GoalMode
import bc_algos.utils.constants as Const
import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import OrderedDict


class SequenceDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract class for fetching sequences of experience. Inherit from this class for 
    different dataset formats. 
    Length of the fetched sequence is equal to (@self.history + @self.action_chunk)
    """
    def __init__(
        self,
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
    ):
        """
        Args:
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
                If GoalMode.LAST, provide last observation as goal for each sequence.
                If GoalMode.SUBGOAL, provide an intermediate observation as goal for each sequence.
                If GoalMode.FULL, provide all subgoals for each sequence.
                Defaults to None, or no goals. 

            num_subgoal (int): (optional) number of subgoals for each demo.
                Defaults to None, which indicates that every frame in the demo is also a subgoal. 
                Assumes that @num_subgoal <= min demo length.

            normalize (bool): if True, normalize data according to mean and stdv 
                computed from the dataset or provided in @normalization_stats.

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset
        """
        self.obs_key_to_modality = obs_key_to_modality
        self.obs_group_to_key = obs_group_to_key
        self.obs_keys = list(set([obs_key for obs_group in obs_group_to_key.values() for obs_key in obs_group]))
        self.action_key = action_key

        assert history >= 0
        self.history = history
        assert action_chunk >= 1
        self.action_chunk = action_chunk
        self.seq_length = history + action_chunk

        self.pad_action_chunk = pad_action_chunk
        self.pad_history = pad_history
        self.get_pad_mask = get_pad_mask

        if goal_mode is not None:
            assert "goal" in obs_group_to_key, "observation group: goal must exist to provide goals"
            assert goal_mode in [GoalMode.LAST, GoalMode.SUBGOAL, GoalMode.FULL], f"goal_mode: {goal_mode} not supported"
            if goal_mode == GoalMode.FULL:
                assert num_subgoal is not None, "goal_mode: full requires the number of subgoals to be specified"
        self.goal_mode = goal_mode
        self.num_subgoal = num_subgoal

        self.normalize = normalize
        self._normalization_stats = normalization_stats
        if normalization_stats is None:
            self._normalization_stats = self.compute_normalization_stats()

        self.cache_index()

    @classmethod
    def factory(cls, config, train=True, normalization_stats=None):
        """
        Create a SequenceDataset instance from config.

        Args:
            config (addict): config object

            train (bool): if True, use kwargs for training dataset. 
                Otherwise, use kwargs for validation dataset.

            normalization_stats (dict): (optional) dictionary from dataset/observation keys to 
                normalization stats from training dataset

        Returns: SequenceDataset instance.
        """
        kwargs = config.dataset.kwargs.train if train else config.dataset.kwargs.valid
        return cls(
            path=config.dataset.path,
            obs_key_to_modality=ObsUtils.OBS_KEY_TO_MODALITY,
            obs_group_to_key=ObsUtils.OBS_GROUP_TO_KEY,
            action_key=config.dataset.action_key,
            history=config.dataset.history,
            action_chunk=config.dataset.action_chunk,
            pad_history=config.dataset.pad_history,
            pad_action_chunk=config.dataset.pad_action_chunk,
            get_pad_mask=config.dataset.get_pad_mask,
            goal_mode=config.dataset.goal_mode,
            num_subgoal=config.dataset.num_subgoal,
            normalize=config.dataset.normalize,
            normalization_stats=normalization_stats,
            **kwargs
        )

    @property
    @abstractmethod
    def demo_ids(self):
        """
        Returns: all demo ids.
        """
        return NotImplementedError      

    @property
    def num_demos(self):
        """
        Returns: number of demos.
        """
        return len(self.demo_ids)

    @property
    def gc(self):
        """
        Returns: whether this dataset contains goals.
        """
        return self.goal_mode is not None
    
    @property
    def normalization_stats(self):
        """
        Returns: if dataset is normalized, a nested dictionary from dataset/observation key 
            to a dictionary that contains keys "mean" and "stdv". Otherwise, None.
        """
        return self._normalization_stats
    
    def compute_normalization_stats(self):
        """
        Compute the mean and stdv of dataset items.

        Returns: nested dictionary from dataset/observation key 
            to a dictionary that contains mean and stdv. 
        """
        traj_dict = {}
        merged_stats = {}

        # don't compute normalization stats for RGB data since we use backbone encoders
        # with their own normalization stats
        keys = [obs_key for obs_key in self.obs_keys if self.obs_key_to_modality[obs_key] != Const.Modality.RGB] + [self.action_key]

        with tqdm(total=self.num_demos, desc="computing normalization stats", unit="demo") as progress_bar:
            for i, demo_id in enumerate(self.demo_ids):
                demo = self.load_demo(demo_id=demo_id)
                traj_dict = {key: demo[key] for key in keys}
                if i == 0:
                    merged_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                else:
                    traj_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                    merged_stats = ObsUtils.aggregate_traj_stats(traj_stats_a=merged_stats, traj_stats_b=traj_stats)

                progress_bar.update(1)
        
        return ObsUtils.compute_normalization_stats(traj_stats=merged_stats, tol=1e-5)
    
    def normalize_demo(self, demo, normalization_stats):
        """
        Normalize @demo in place according to @normalization_stats.

        Args: 
            demo (dict): nested dictionary returned from @self.load_demo

            normalization_stats (dict): nested dictionary from dataset/observation key 
                to a dictionary that contains mean and stdv. 
        """
        for key in normalization_stats:
            if key in demo: demo[key] = ObsUtils.normalize(data=demo[key], normalization_stats=normalization_stats[key])

    @abstractmethod
    def demo_len(self, demo_id):
        """
        Args: 
            demo_id: demo id
        
        Returns: length of demo with @demo_id.
        """
        return NotImplementedError

    @abstractmethod
    def _fetch_demo(self, demo_id):
        """
        Fetch demo with @demo_id from memory.

        Args: 
            demo_id: demo id

        Returns: nested dictionary with the following format:
        {
            dataset_key: data (np.array) of shape [T, ...]
            ...
            obs_key: data (np.array) of shape [T, ...]
            ...
        }
        """
        return NotImplementedError
    
    def load_demo(self, demo_id):
        """
        Fetch demo with @demo_id from memory and preprocess it.

        Args: 
            demo_id: demo id

        Returns: nested dictionary with the following format:
        {
            dataset_key: data (np.array) of shape [T, ...]
            ...
            obs_key: data (np.array) of shape [T, ...]
            ...
        }
        """
        demo = self._fetch_demo(demo_id=demo_id)
        if self.normalize:
            self.normalize_demo(demo=demo, normalization_stats=self._normalization_stats)
        return demo
    
    def cache_index(self):
        """
        Cache all indices required for get_item calls to speed up training.
        """
        self.index_cache = {}
        with tqdm(total=self.num_demos, desc="caching sequences", unit="demo") as progress_bar:
            for demo_id in self.demo_ids:
                demo_length = self.demo_len(demo_id=demo_id)

                data_index = np.arange(demo_length)
                if self.pad_history:
                    data_index = np.append(np.zeros(self.history), data_index)
                if self.pad_action_chunk:
                    data_index = np.append(data_index, np.full(self.action_chunk-1, -1))
                data_index = data_index.astype(np.int32)

                pad_index = None
                if self.get_pad_mask:
                    pad_index = np.full(demo_length, 1)
                    if self.pad_history:
                        pad_index = np.append(np.zeros(self.history), pad_index)
                    if self.pad_action_chunk:
                        pad_index = np.append(pad_index, np.zeros(self.action_chunk-1))
                    pad_index = pad_index.astype(np.int32)
                
                goal_index = None
                if self.goal_mode == GoalMode.LAST:
                    goal_index = np.array([-1])
                elif self.goal_mode == GoalMode.SUBGOAL:
                    if self.num_subgoal is None:
                        goal_index = np.arange(1, demo_length+1)
                    else:
                        subgoal_index = np.linspace(0, demo_length, self.num_subgoal+1, dtype=np.uint32)
                        goal_index = np.repeat(subgoal_index[1:], np.diff(subgoal_index))
                elif self.goal_mode == GoalMode.FULL:
                    goal_index = np.linspace(0, demo_length, self.num_subgoal+1, dtype=np.uint32)[1:]

                self.index_cache[demo_id] = (data_index, pad_index, goal_index)

                progress_bar.update(1)

    def extract_data_seq(self, demo, keys, seq_index):
        """
        Extract a (sub)sequence from @demo.

        Args:
            demo (dict): nested dictionary returned from @self.load_demo

            keys (array): keys to extract

            seq_index (array): sequence indices

        Returns: nested dictionary from dataset/observation key to
            data (np.array) of shape [T, ...]
        """
        data_seq = OrderedDict()
        for k in keys:
            data_seq[k] = demo[k][seq_index]
        return data_seq
    
    def seq_from_timstep(self, demo_id, demo, t):
        """
        Get sequence for timestep @t in @demo.

        Args: 
            demo_id: demo id

            demo (dict): nested dictionary returned from @self.load_demo

            t (int): timestep in demo

        Returns: nested dictionary with three possible items:
        
            1) obs (dict): dictionary from observation key to data (np.array)
                of shape [T = @self.history + @self.action_chunk, ...]
            
            2) goal (dict): dictionary from observation key to data (np.array) of shape [T_goal, ...]

            3) pad_mask (np.array): mask of shape [T = @self.history + @self.action_chunk] 
                indicating which frames are padding
        """
        data_index, pad_index, goal_index = self.index_cache[demo_id]
        data_seq_index = data_index[t:t+self.seq_length]
        seq = self.extract_data_seq(demo=demo, keys=[self.action_key], seq_index=data_seq_index)
        seq["obs"] = self.extract_data_seq(demo=demo, keys=self.obs_group_to_key["obs"], seq_index=data_seq_index)
        if self.gc:
            goal_seq_index = goal_index
            if self.goal_mode == GoalMode.SUBGOAL:
                goal_seq_index = goal_index[t+self.history:t+self.seq_length]
            seq["goal"] = self.extract_data_seq(demo=demo, keys=self.obs_group_to_key["goal"], seq_index=goal_seq_index)
        if self.get_pad_mask:
            seq["pad_mask"] = pad_index[t:t+self.seq_length]
        return seq
    
    def __getitem__(self, index):
        """
        Randomly sample a sequence from a demo. 
        """
        demo_id = self.demo_ids[index]
        demo = self.load_demo(demo_id=demo_id)
        sample_size = self.demo_len(demo_id=demo_id)
        if not self.pad_history:
            sample_size -= self.history
        if not self.pad_action_chunk:
            sample_size -= (self.action_chunk-1) 
        t = np.random.choice(sample_size)
        return self.seq_from_timstep(demo_id=demo_id, demo=demo, t=t)
    
    def __len__(self):
        """
        A full pass through dataset consists of sampling a sequence from each demo.
        """
        return self.num_demos
    
    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = "\thistory={}\n\taction_chunk={}\n\tpad_history={}\n\tpad_action_chunk={}\n"
        msg += "\tnum_demos={}\n"
        msg += "\tgoal_mode={}\n\tnum_subgoal={}\n"
        msg += "\tnormalize={}\n"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        num_subgoal_str = self.num_subgoal if self.num_subgoal is not None else "none"
        msg = msg.format(
            self.history, self.action_chunk, self.pad_history, self.pad_action_chunk,
            self.num_demos,
            goal_mode_str, num_subgoal_str,
            self.normalize,
        )
        return msg
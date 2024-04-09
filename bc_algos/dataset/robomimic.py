"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
from bc_algos.dataset.dataset import SequenceDataset
import h5py
import numpy as np
from tqdm import tqdm
import bc_algos.utils.constants as Const
import bc_algos.utils.obs_utils as ObsUtils
from collections import OrderedDict


class RobomimicDataset(SequenceDataset):
    """
    Class for fetching sequences of experience from Robomimic dataset.
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
        filter_by_attribute=None,
        demos=None,
        preprocess=False,
        normalize=False,
    ):
        """
        SequenceDataset subclass for fetching sequences of experience from HDF5 dataset.

        Args:
            path (str): path to dataset

            obs_key_to_modality (dict): dictionary from observation key to modality

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

            goal_mode (str): either "last", "subgoal", or None. Defaults to None, or no goals

            num_subgoal (int): Required if goal_mode is "subgoal". Number of subgoals provided for each trajectory.
                Defaults to None, which indicates that every state is also a subgoal. Assume num_subgoal <= min length of traj.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            demos (list): if provided, only load these selected demos

            preprocess (bool): if True, preprocess data while loading into memory

            normalize (bool): if True, normalize data using mean and stdv from dataset
        """
        self._hdf5_file = None
        self.filter_by_attribute = filter_by_attribute

        super(RobomimicDataset, self).__init__(
            path=path,
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
            demos=demos,
            preprocess=preprocess,
            normalize=normalize,
        )

    @property
    def demos(self):
        """
        Returns: all demo ids.
        """
        if self._demos is not None:
            return self._demos
        demos = []
        # filter demo trajectory by mask
        if self.filter_by_attribute is not None:
            demos = [elem.decode("utf-8") for elem in self.hdf5_file[f"mask/{self.filter_by_attribute}"][:]]
        else:
            demos = list(self.hdf5_file["data"].keys())
        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        return demos

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.path, 'r', swmr=True, libver='latest')
        return self._hdf5_file   

    def demo_len(self, demo_id):
        """
        Args: 
            demo_id: demo id, ie. "demo_0"
        
        Returns: length of demo with @demo_id.
        """
        return self.hdf5_file[f"data/{demo_id}"].attrs["num_samples"] 
    
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

    def load_dataset_in_memory(self, preprocess):
        """
        Load the dataset into memory.

        Args: 
            preprocess (bool): if True, preprocess data while loading into memory
        """
        dataset = dict()

        with tqdm(total=self.num_demos, desc="loading dataset into memory", unit='demo') as progress_bar:
            for demo in self.demos:
                dataset[demo] = dict()

                # get observations
                dataset[demo] = {obs_key: self.hdf5_file[f"data/{demo}/obs/{obs_key}"][()] for obs_key in self.obs_keys}
                if preprocess:
                    for obs_key in self.obs_keys:
                        if self.obs_key_to_modality[obs_key] == Const.Modality.RGB:
                            dataset[demo][obs_key] = ObsUtils.preprocess_img(img=dataset[demo]["obs"][obs_key])

                # get other dataset keys
                for dataset_key in self.dataset_keys:
                    dataset[demo][dataset_key] = self.hdf5_file[f"data/{demo}/{dataset_key}"][()].astype('float32')

                progress_bar.update(1)

        self.dataset = dataset

    def compute_normalization_stats(self):
        """
        Compute the mean and stdv for dataset items and store stats at @self.normalization_stats.
        The format for @self.normalization_stats should be a dictionary from dataset/observation
        key to a dictionary that contains mean and stdv. 

        Example:
        {
            "actions": {
                "mean": ...,
                "stdv": ...
            }
        }
        """
        traj_dict = dict()
        merged_stats = dict()

        # don't compute normalization stats for RGB data since we use backbone encoders
        # with their own normalization stats
        keys = [obs_key for obs_key in self.obs_keys if self.obs_key_to_modality[obs_key] != Const.Modality.RGB] + self.dataset_keys

        with tqdm(total=self.num_demos, desc="computing normalization stats", unit="demo") as progress_bar:
            for i, demo in enumerate(self.demos):
                traj_dict = {key: self.dataset[demo][key] for key in keys}
                if i == 0:
                    merged_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                else:
                    traj_stats = ObsUtils.compute_traj_stats(traj_dict=traj_dict)
                    merged_stats = ObsUtils.aggregate_traj_stats(traj_stats_a=merged_stats, traj_stats_b=traj_stats)

                progress_bar.update(1)
        
        self.normalization_stats = ObsUtils.compute_normalization_stats(traj_stats=merged_stats)

    def normalize_data(self):
        """
        Normalize dataset items according to @self.normalization_stats.
        """
        with tqdm(total=self.num_demos, desc="normalizing data", unit="demo") as progress_bar:
            for demo in self.demos:
                for key in self.normalization_stats:
                    self.dataset[demo][key] = ObsUtils.normalize(data=self.dataset[demo][key], normalization_stats=self.normalization_stats[key])
                    
                progress_bar.update(1)
    
    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of dataset items from a demo.

        Args:
            demo_id: demo id, ie. "demo_0"

            keys (tuple): keys to extract

            seq_index (list): sequence indices

        Returns: ordered dictionary of extracted items.
        """
        # fetch observation from the dataset file
        seq = OrderedDict()
        for k in keys:
            data = self.dataset[demo_id][k]
            seq[k] = data[seq_index]
        return seq
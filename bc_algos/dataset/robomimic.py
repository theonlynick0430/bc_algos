"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
from bc_algos.dataset.dataset import MIMODataset
import h5py
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm


class RobomimicDataset(MIMODataset):
    """
    Class for fetching sequences of experience from Robomimic dataset.
    Length of the fetched sequence is equal to (@frame_stack + @seq_length)
    """
    def __init__(
        self,
        path,
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
    ):
        """
        MIMO_Dataset subclass for fetching sequences of experience from HDF5 dataset.

        Args:
            path (str): path to dataset

            obs_group_to_key (dict): dictionary from observation group to observation key

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 0 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last", "subgoal", or None. Defaults to None, which is to not fetch goals

            num_subgoal (int): Required if goal_mode is "subgoal". Number of subgoals provided for each trajectory.
                Defaults to None, which indicates that every state is also a subgoal. Assume num_subgoal <= min length of traj.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            demos (list): if provided, use only load these selected demos
        """
        self._hdf5_file = None
        self.filter_by_attribute = filter_by_attribute

        super(RobomimicDataset, self).__init__(
            path=path,
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
        )

        self.dataset = self.load_dataset_in_memory()

        self.close_and_delete_hdf5_handle()

    def load_dataset_in_memory(self):
        """
        Load the hdf5 dataset into memory, preserving the structure of the file.

        Returns:
            dataset (dict): dictionary of loaded data
        """
        print("loading dataset into memory...")
        dataset = dict()
        with tqdm(total=self.num_demos, unit='demo') as progress_bar:
            for demo in self.demos:
                dataset[demo] = {}

                # get observations
                dataset[demo]["obs"] = {}
                for obs_key in self.obs_keys:
                    assert obs_key in self.hdf5_file[f"data/{demo}/obs"], f"obs_key {obs_key} not found in dataset"
                    dataset[demo]["obs"][obs_key] = self.hdf5_file[f"data/{demo}/obs/{obs_key}"][()].astype('float32')

                # get other dataset keys
                for dataset_key in self.dataset_keys:
                    assert dataset_key in self.hdf5_file[f"data/{demo}"], f"dataset_key {dataset_key} not found in dataset"
                    dataset[demo][dataset_key] = self.hdf5_file[f"data/{demo}/{dataset_key}"][()].astype('float32')

                progress_bar.update(1)

        return dataset

    @property
    def demos(self):
        """
        Get all demo ids
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
    def num_demos(self):
        """
        Get number of demos
        """
        return len(self.demos)   

    def get_demo_len(self, demo_id):
        """
        Get length of demo with demo_id
        """
        return self.hdf5_file[f"data/{demo_id}"].attrs["num_samples"] 

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.path, 'r', swmr=True, libver='latest')
        return self._hdf5_file  
    
    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
    
    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()
    
    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_group_to_key={}\n\tobs_keys={}\n\tfilter_key={}\n"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        msg = msg.format(self.path, self.obs_group_to_key, self.obs_keys, filter_key_str)
        return msg + super(RobomimicDataset, self).__repr__() + ")"
    
    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (tuple): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.dataset[demo_id][k]
            seq[k] = data[seq_index]

        return seq
    
    def get_obs_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (tuple): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        seq = dict()
        for k in keys:
            data = self.dataset[demo_id]["obs"][k]
            seq[k] = data[seq_index]
        return seq
"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
from bc_algos.dataset.dataset import MIMODataset
import os
import h5py
import numpy as np
from contextlib import contextmanager


class RobomimicDataset(MIMODataset):
    """
    Class for fetching sequences of experience from Robomimic dataset.
    """

    def __init__(
        self,
        hdf5_path,
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
            hdf5_path (str): path to hdf5.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load
        """
        self.hdf5_path = os.path.expanduser(hdf5_path)
        self._hdf5_file = None
        self.filter_by_attribute = filter_by_attribute
        self._demos = demos

        super(RobomimicDataset, self).__init__(
            obs_group_to_key=obs_group_to_key,
            dataset_keys=dataset_keys,
            frame_stack=frame_stack,
            seq_length=seq_length, 
            pad_frame_stack=pad_frame_stack, 
            pad_seq_length=pad_seq_length, 
            get_pad_mask=get_pad_mask, 
            goal_mode=goal_mode, 
            num_subgoal=num_subgoal
            )

        self.close_and_delete_hdf5_handle()

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
            demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(self.filter_by_attribute)][:])]
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
        return self.hdf5_file["data/{}".format(demo_id)].attrs["num_samples"] 

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True, libver='latest')
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
        msg = msg.format(self.hdf5_path, self.obs_group_to_key, self.obs_keys, filter_key_str)
        return msg + super(RobomimicDataset, self).__repr__() + ")"

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        """
        hd5key = "data/{}/{}".format(ep, key)
        return self.hdf5_file[hd5key]
    
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
            data = self.get_dataset_for_ep(demo_id, k)
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
        seq = self.get_data_seq(
            demo_id=demo_id,
            keys=['{}/{}'.format("obs", k) for k in keys], 
            seq_index=seq_index
            )
        seq = {k.split('/')[1]: seq[k] for k in seq}  # strip the prefix
        return seq
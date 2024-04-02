"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from pickled files.
"""
from bc_algos.dataset.dataset import MIMODataset
from tqdm import tqdm
import os
from bc_algos.utils.misc import load_gzip_pickle
import numpy as np
from PIL import Image


class IsaacGymDataset(MIMODataset):
    """
    Class for fetching sequences of experience from Isaac Gym dataset.
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
        demos=None,
    ):
        """
        MIMO_Dataset subclass for fetching sequences of experience from HDF5 dataset.

        Args:
            path (str): path to dataset folder

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

            demos (list): if provided, use only load these selected demos
        """
        super(IsaacGymDataset, self).__init__(
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
                demo_len = self.get_demo_len(demo)

                dataset[demo] = {dataset_key: [] for dataset_key in self.dataset_keys}
                dataset[demo]["obs"] = {obs_key: [] for obs_key in self.obs_keys}

                for step in range(1, demo_len+1):
                    state_path = os.path.join(self.path, f"run_{demo}/state_{step}.pkl.gzip")
                    state = load_gzip_pickle(filename=state_path)

                    # get observations
                    for obs_key in self.obs_keys:
                        assert obs_key in state, f"obs_key {obs_key} not found in dataset"
                        dataset[demo]["obs"][obs_key].append(state[obs_key].astype('float32'))

                    # TODO: include img obs in pickled files
                    # assume for now image obs stored in field "agentview_image"
                    img_path = os.path.join(self.path, f"run_{demo}/im_{step}.png")
                    img = (np.array(Image.open(img_path))/255.).transpose(2, 0, 1)
                    dataset[demo]["obs"]["agentview_image"].append(img)

                    # get observations
                    for dataset_key in self.dataset_keys:
                        assert dataset_key in state, f"dataset_key {dataset_key} not found in dataset"
                        dataset[demo][dataset_key].append(state[dataset_key].astype('float32'))

                # convert to np arrays
                dataset[demo] = {dataset_key: np.array([dataset[demo][dataset_key]]) for dataset_key in self.dataset_keys}
                dataset[demo]["obs"] = {obs_key: np.array([dataset[demo]["obs"][obs_key]]) for obs_key in self.obs_keys}

                progress_bar.update(1)

        return dataset

    @property
    def demos(self):
        """
        Get all demo ids
        """
        if self._demos is not None:
            return self._demos
        demos = [i for i in range(1, len(os.listdir(self.path)+1))]
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
        meta_path = os.path.join(self.path, f"run_{demo_id}/metadata.pkl.gzip")
        meta = load_gzip_pickle(filename=meta_path)
        return meta["num_steps"] 

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_group_to_key={}\n\tobs_keys={}\n"
        msg = msg.format(self.path, self.obs_group_to_key, self.obs_keys)
        return msg + super(IsaacGymDataset, self).__repr__() + ")"
    
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
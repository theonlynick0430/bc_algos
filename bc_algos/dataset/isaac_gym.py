"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from pickled files.
"""
from bc_algos.dataset.dataset import SequenceDataset
import bc_algos.utils.constants as Const
import bc_algos.utils.obs_utils as ObsUtils
from tqdm import tqdm
import os
from bc_algos.utils.misc import load_gzip_pickle
import numpy as np
from PIL import Image


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
        preprocess=False,
        normalize=True,
    ):
        """
        SequenceDataset subclass for fetching sequences of experience from HDF5 dataset.

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

            goal_mode (GoalMode): either GoalMode.LAST, GoalMode.SUBGOAL, GoalMode.FULL, or None. 
                If GoalMode.LAST, provide last observation as goal.
                If GoalMode.SUBGOAL, provide an intermediate observation as goal for each frame in sampled sequence.
                If GoalMode.FULL, provide all subgoals for a single batch.
                Defaults to None, or no goals. 

            num_subgoal (int): (optional) number of subgoals for each trajectory.
                Defaults to None, which indicates that every frame in trajectory is also a subgoal. 
                Assumes that @num_subgoal <= min trajectory length.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            demos (array): if provided, only load these selected demos

            preprocess (bool): if True, preprocess data while loading into memory

            normalize (bool): if True, normalize data using mean and stdv from dataset
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
            preprocess=preprocess,
            normalize=normalize,
        )

    @property
    def demos(self):
        """
        Get all demo ids
        """
        if self._demos is not None:
            return self._demos
        return [i for i in range(len(os.listdir(self.path)))]

    @property
    def demos(self):
        """
        Returns: all demo ids.
        """
        if self._demos is None:
            if self.filter_by_attribute is not None:
                if self.filter_by_attribute == "train":

                self._demos = [elem.decode("utf-8") for elem in self.hdf5_file[f"mask/{self.filter_by_attribute}"][:]]
            else:
                self._demos = list(self.hdf5_file["data"].keys())
        return self._demos

    def get_demo_len(self, demo_id):
        """
        Get length of demo with demo_id
        """
        run_path = os.path.join(self.path, f"run_{demo_id}.pkl.gzip")
        run = load_gzip_pickle(filename=run_path)
        return run["metadata"]["num_steps"]

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__) + "(\n"
        msg += super(IsaacGymDataset, self).__repr__() + ")"
        return msg

    def load_dataset_in_memory(self, preprocess):
        """
        Load the dataset into memory.

        Args: 
            preprocess (bool): if True, preprocess data while loading into memory
        """
        dataset = {}

        with tqdm(total=self.num_demos, desc="loading dataset into memory", unit='demo') as progress_bar:
            for demo in self.demos:
                dataset[demo] = {}

                run_path = os.path.join(self.path, f"run_{demo}.pkl.gzip")
                run = load_gzip_pickle(run_path)

                # get observations
                dataset[demo] = {obs_key: run["obs"][obs_key] for obs_key in self.obs_keys}
                if preprocess:
                    for obs_key in self.obs_keys:
                        if self.obs_key_to_modality[obs_key] == Const.Modality.RGB:
                            dataset[demo][obs_key] = ObsUtils.preprocess_img(img=dataset[demo][obs_key])

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
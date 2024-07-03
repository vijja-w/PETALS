import os
import random
import numpy as np

import torch
torch.set_float32_matmul_precision('high')

from general.utils import set_seed, ImageDataset
from method_contrast.models import CNN
from pydantic import BaseModel


def model_init(config):
    model_name = config['model_name']
    set_seed(config['seed'])
    if model_name == 'cnn':
        model = CNN(config)
    else:
        raise NotImplementedError(f'model_name ({model_name}) not implemented')
    return model


def data_init(mode, config):
    dataloader = CDataLoader(mode, config)
    return dataloader


class CDataLoader:
    """
    A data loader class for contrastive learning constructed from the 'ImageDataset'.

    This class is used to load and prepare data for contrastive learning tasks based on the 'ImageDataset' object.
    The 'ImageDataset' is characterised by 'ns' unique samples referred to using the index 's'. Each unique sample
    has 'nt' time samples which are temporally ordered and referred to using the time index 't'. The batch size is
    equal to 'ns_batch' * 'nt_batch'. Going through one epoch means going through all 'ns' unique samples. As an
    example if ns = 10 and ns_batch = 2, then five batches will be considered an epoch. Each time an 's' is selected
    to be in a batch, 'nt_batch' time samples for each 's' is sampled to build the batch. When sampling 't'samples,
    we obtain the anchor, positives and negatives. The number of positive and negative sampels to sample is
    determined by 'p_pos'. There is only one anchor. The number of positives, negatives and anchor sums up to
    'nt_batch'. How big the region in the temporal sequence around the anchor that is considered positives is
    determined by 'p_pos_window'.

    Args:
        mode (str):
            Dataset to load. mode in ('train', 'val', 'test').
        config (dict): A dictionary containing configuration parameters for the model.
            - dataset (ImageDataset): The dataset containing image data and associated information.
            - ns_batch (int): The size of the source batch.
            - nt_batch (int): The size of the target batch.
            - p_pos (float): The positive value for contrastive learning.
            - p_pos_window (float): The positive window value for defining positive indices.
            - shuffle (bool): Determines whether to shuffle the 's' indices when iterating through the dataset.
        verbose (bool):
            Print information when instantiated.

    Methods:
        check_signature(config):
            Checks that config has all the keys required and that they are all of the correct type.

        set_batch_params(ns_batch, nt_batch, p_pos, p_pos_window):
            Set batch parameters for the data loader.

        partition_indices(s, anchor_index):
            Partition indices around an anchor index for a specific sample 's'.

        sample_batch_indices(s):
            Sample batch indices for a specific sample 's'.

        vis_batch():
            Visualize a batch of sequences with annotations.
    """

    class Config(BaseModel):
        data_dir: str = os.path.join('data', 'lettuce')
        ns_batch: int = 8
        nt_batch: int = 13
        p_pos: float = 0.2
        p_pos_window: float = 0.125
        shuffle: bool = True
        verbose: bool = False
    default_config = Config().dict()

    def __init__(self, mode: str, config: dict):
        _ = self.Config(**config).dict()

        # attributes
        self.data_dir = config['data_dir']
        self.ns_batch = config['ns_batch']
        self.nt_batch = config['nt_batch']
        self.p_pos = config['p_pos']
        self.p_pos_window = config['p_pos_window']
        self.shuffle = config['shuffle']
        self.verbose = config['verbose']

        dataset = ImageDataset(self.data_dir, mode)
        if dataset.mode in ['val', 'test'] and config['ns_batch'] > dataset.ns:
            if self.verbose:
                print(f'ns_batch ({self.ns_batch}) > dataset.ns ({dataset.ns}) but mode ({dataset.mode}) is val or test.'
                      f'ns_batch has therefore been set to dataset.ns automatically.')
            self.ns_batch = dataset.ns

        self.dataset = dataset
        self.n_pos = round(self.p_pos * self.nt_batch)
        self.n_neg = self.nt_batch - self.n_pos
        self.batch_size = self.ns_batch * self.nt_batch
        self.num_samples = dataset.ns
        self.set_batch_params(self.ns_batch, self.nt_batch, self.p_pos, self.p_pos_window)

        min_nt = min(dataset.s2nt.values())
        min_n_pos_window = round(self.p_pos_window * min_nt) - 1     # -1 from anchor image itself
        if self.verbose:
            if not all(nt*0.75 <= self.nt_batch for nt in dataset.s2nt.values()):
                print('nt_batch is more than 75% of the data in some s in the dataset, might be too large.')

            print('---------------------------------------------------------------------------------------------------')
            print('Check the following are reasonable:')
            print('---------------------------------------------------------------------------------------------------')
            print(f'Number of positives per unique sample in a batch: {self.n_pos}')
            print(f'Number of negatives per unique sample in a batch: {self.n_neg}')
            print(f'Minimum number of samples considered as positives around anchor: {min_n_pos_window}')
            print(f'The minimum nt for a unique sample: {min_nt}')
            print('')
            print('In a batch:')
            print(f'  - Batch size: {self.batch_size}')
            print(f'  - Number of unique samples: {self.ns_batch}')
            print(f'  - Number of samples per unique sample: {self.nt_batch}')
            print('')
            print('Batch sampling looks like this:')
            vis_batch = self.vis_batch()
            for b in vis_batch:
                print(b)
            print('')
            print('---------------------------------------------------------------------------------------------------')

    def __iter__(self):
        """
        Iterate through batches of data samples.

        Generates the batches according to the description in the Class docstring.

        Yields:
            tuple: A tuple containing two elements:
                - batch (list): A list of data samples in the batch.
                - batch_labels (list): A list of labels associated with the data samples in the batch for debugging.

        """

        if self.shuffle:
            s_indices = random.sample(self.dataset.s_list, len(self.dataset.s_list))
        else:
            s_indices = self.dataset.s_list

        # Iterate through the list of sample indices in batches of size 'ns_batch'
        for start_idx in range(0, len(s_indices), self.ns_batch):
            end_idx = min(start_idx + self.ns_batch, len(s_indices))
            s_indices_batch = s_indices[start_idx:end_idx]

            batch = []
            batch_labels = []
            batch_labels_raw = []

            # Iterate through the 's' indices to be included in the batch
            for s in s_indices_batch:
                # Sample positive, negative and anchor indices for the current unique sample 's'
                sampled_pos_indices, sampled_neg_indices, anchor_index, _, _ = self.sample_batch_indices(s)

                # Create labels for the current batch based on sample, anchor, and positive/negative indices
                batch_labels.append(0)
                batch_labels_raw.append(f's{s}a{anchor_index}')
                batch.append(self.dataset.get_item(anchor_index, s))

                for t_pos in sampled_pos_indices:
                    batch_labels.append(1)
                    batch_labels_raw.append(f's{s}p{t_pos}')
                    batch.append(self.dataset.get_item(t_pos, s))

                for t_neg in sampled_neg_indices:
                    batch_labels.append(2)
                    batch_labels_raw.append(f's{s}n{t_neg}')
                    batch.append(self.dataset.get_item(t_neg, s))

            yield torch.cat(batch, dim=0), (torch.tensor(batch_labels, dtype=torch.float32), batch_labels_raw)

    def set_batch_params(self, ns_batch: int, nt_batch: int, p_pos: float, p_pos_window: float):
        """
        Set batch parameters. Useful for using with the vis_batch method to experiment with batch parameters and
        visualise it. Resulting batch size is ns_batch * nt_batch.

        Args:
            ns_batch (int): The number of unique samples 's' to use in the batch.
            nt_batch (int): The number of time samples 't' to include for each unique sample 's'.
            p_pos (float): Percentage (between 0 and 1) of nt_batch to be used as positive samples for the anchor.
            p_pos_window (float): Percentage (between 0 and 1) of nt_batch to be considered as positives around
                the anchor.

        Raises:
            ValueError: If any of the following conditions are not met:
                - `ns_batch` is greater than the number of unique samples 's' in the provided dataset.
                - `nt_batch` is too large for some 's' in the dataset.
                - `ns_batch` is less than 1.
                - `nt_batch` is less than 1.
                - `p_pos` is not between 0 and 1.
                - `p_pos_window` is not between 0 and 1.
                - The minimum number of samples considered as positives around the anchor is less than the number
                  of positives required.

        Returns:
            None
        """

        if self.dataset.ns < ns_batch:
            raise ValueError('n_s_batch not less than n_s in the provided dataset')

        if any(nt_batch >= nt for nt in self.dataset.s2nt.values()):
            raise ValueError('nt_batch specified is too large for some s in the dataset')

        if ns_batch < 1:
            raise ValueError('ns_batch has to be greater or equal to 1')
        if nt_batch < 1:
            raise ValueError('nt_batch has to be greater or equal to 1')

        if not 0 <= p_pos <= 1:
            raise ValueError("p_pos should be between 0 and 1")

        if not 0 <= p_pos_window <= 1:
            raise ValueError("p_pos_window should be between 0 and 1")

        if round(p_pos_window * min(self.dataset.s2nt.values())) - 1 <= self.n_pos:
            raise ValueError('Minimum number of samples considered as positives around anchor is less than '
                             'the number of positives required.')

        self.ns_batch = ns_batch
        self.nt_batch = nt_batch
        self.p_pos = p_pos
        self.p_pos_window = p_pos_window

    def partition_indices(self, s: int, anchor_index: int):
        """
        Partition indices around an anchor index for a specific sample 's'.

        This method calculates positive and negative indices around an anchor index for a specific sample 's' based on
        the positive window size (`p_pos_window`). It returns two lists: positive indices and negative indices.

        Args:
            s (int):  The index of the unique sample.
            anchor_index (int): The anchor index around which to partition indices.

        Returns:
            tuple: A tuple containing two lists:
                - pos_indices (list): Positive indices around the anchor index. DOES NOT CONTAIN THE ANCHOR INDEX.
                - neg_indices (list): Remaining indices considered as negatives.

        """
        nt = self.dataset.s2nt[str(s)]      # Get the total number of time samples for sample 's'
        n_pos_window = round(self.p_pos_window * nt)    # Calculate the size of the positive window

        # Calculate positive indices around the anchor index
        pos_indices = np.array([anchor_index + i - (n_pos_window//2) for i in range(n_pos_window)])

        # Adjust positive indices to handle edge cases
        if pos_indices[0] < 0:
            pos_indices -= pos_indices[0]
        elif pos_indices[-1] >= nt:
            pos_indices -= pos_indices[-1] - nt + 1

        # Calculate negative indices as the complement of positive indices
        neg_indices = list(set(range(nt)) - set(pos_indices))

        # Remove the anchor index from the positive indices
        pos_indices = list(pos_indices)
        pos_indices.remove(anchor_index)
        return pos_indices, neg_indices

    def sample_batch_indices(self, s: int):
        """
        Sample batch indices for a unique sample with index 's'.

        This method samples positive and negative indices for a unique sample with index 's' using a random anchor index
        and the positive/negative sample partitioning defined by the `partition_indices` method. It then samples from
        this and returns the sampled positive indices, sampled negative indices, the anchor index, partitioned positive
        and negative indices.

        Args:
            s (int): The index of the unique sample for which indices are sampled.

        Returns:
            tuple: A tuple containing the following elements:
                - sampled_pos_indices (list): Sampled positive indices around the anchor index.
                - sampled_neg_indices (list): Sampled negative indices far from the anchor index.
                - anchor_index (int): The randomly chosen anchor index.
                - pos_indices (list): Positive indices around the anchor index for the particular 's'.
                - neg_indices (list): Negative indices far from the the anchor index for the particular 's'.
        """

        nt = self.dataset.s2nt[str(s)]   # Get the total number of time samples for sample 's'

        # Choose a random anchor index within the range of time samples 'nt'
        anchor_index = random.randint(0, nt - 1)

        # Use the `partition_indices` method to get the original positive and negative indices
        pos_indices, neg_indices = self.partition_indices(s, anchor_index)

        # Sample a specified number of positive and negative indices randomly
        sampled_pos_indices = random.sample(pos_indices, self.n_pos)
        sampled_neg_indices = random.sample(neg_indices, self.n_neg)

        return sampled_pos_indices, sampled_neg_indices, anchor_index, pos_indices, neg_indices

    def vis_batch(self):
        """
        Visualize a batch sampled batch using a sequence of strings.

        This method generates a potential batch and visualizes it with annotations.

        The 'ns_batch' unique samples used are sampled randomly without replacement here to get the 's' indices. These
        are then used to generate the batch. Each newline corresponds to an 's' index.

        Each line corresponds to a partitioning and sampling for each unique sample 's', with 'A' for the anchor
        index, 'N' for sampled negative indices, 'n' for the other indices that are considered negative by the
        partitioning, 'P' for sampled positive indices, and 'p' for other indices that are considered positive by the
        partitioning. The batch size is equal to `ns_batch` * `nt_batch`.

        Returns:
            list: A list of strings representing the sampled batch.

        """

        vis_batch = []

        # Sample 'ns_batch' random 's' and iterate through these
        for s in random.sample(list(range(self.dataset.ns)), self.ns_batch):
            sampled_pos_indices, sampled_neg_indices, anchor_index, pos_indices, neg_indices = self.sample_batch_indices(s)

            # Create a character mapping that encodes the representation
            char_mapping = {anchor_index: 'A',
                            **{i: 'N' if i in sampled_neg_indices else 'n' for i in neg_indices},
                            **{i: 'P' if i in sampled_pos_indices else 'p' for i in pos_indices}}

            # Generate the strings
            b = ''.join(char_mapping.get(i, '') for i in range(self.dataset.s2nt[str(s)]))
            vis_batch.append('   ' + b)

        return vis_batch

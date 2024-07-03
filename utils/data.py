from copy import deepcopy
from pydantic import BaseModel

import os
import json
import random

from tqdm import tqdm
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import plotly


def init_dataloader(config, mode):
    data_name2dataloader = {
        't': TDataLoader,
        'o': ODataLoader,
        'dr': DRDataLoader,
        'c': CDataLoader
    }
    data_name = config['data_name']
    if data_name in data_name2dataloader:
        return data_name2dataloader[data_name](mode, config)
    else:
        raise NotImplementedError(f"data_name ({data_name}) not in {data_name2dataloader.keys()}.")


class ImageDataset(Dataset):
    def __init__(self, data_dir=os.path.join('data', 'lettuce'), mode: str = 'train'):
        self.data_dir = data_dir
        self.mode = mode

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid mode. Supported values are 'train', 'val', 'test'.")

        with open(os.path.join(data_dir, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)
        self.extension = metadata['extension']
        self.s2nt = metadata['s2nt']
        self.s_list = metadata[mode]
        self.ns = len(self.s_list)
        self.s2t_list = defaultdict(list)

        self.image_paths = []
        for s in self.s_list:
            for t in sorted([int(f.split('t')[1].split(f'.{self.extension}')[0]) for f in os.listdir(data_dir + f'/s{s}')]):
                self.s2t_list[s].append(t)
                self.image_paths.append(os.path.join(data_dir, f's{s}', f't{t}.{self.extension}'))
        self.len_dataset = len(self.image_paths)

    def get_item(self, t, s):
        if s not in self.s_list:
            raise ValueError(f"Requested 's' ({s}) not in dataset for mode={self.mode}")
        image_path = os.path.join(self.data_dir, f's{s}', f't{t}.{self.extension}')
        image = Image.open(image_path)
        return self.transforms(image).unsqueeze(0)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx])


class TDataLoader:
    """
    Generates the dataloader to generate batches of (x, t), where x is the image and t is the wall-clock age.
    """

    class Config(BaseModel):
        data_dir: str = os.path.join('data', 'lettuce')
        batch_size: int = 16
        shuffle: bool = True
    default_config = Config().dict()

    def __init__(self, mode: str, config: dict):
        _ = self.Config(**config).dict()

        # attributes
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.data_dir = config['data_dir']

        # init dataset
        self.dataset = ImageDataset(self.data_dir, mode)

        # generate list of s, t pairs.
        self.st_list = []
        for s in self.dataset.s_list:
            for t in range(self.dataset.s2nt[str(s)]):
                self.st_list.append([s, t])
        self.st_list = np.array(self.st_list)

    def __iter__(self):
        """
        Iterate through batches of data samples.

        Generates the batches according to the description in the Class docstring.

        Yields:
            tuple: A tuple containing two elements:
                - batch (list): A list of image samples (x) in the batch.
                - batch_labels (list): A list of labels (t) associated with the data samples in the batch.
        """

        st_list = self.st_list.copy()
        if self.shuffle:
            np.random.shuffle(st_list)

        # split st into batches
        for start_idx in range(0, len(st_list), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(st_list))
            st_batch = st_list[start_idx:end_idx]

            # generate batch
            batch_x = []
            batch_labels = []
            for s, t in st_batch:
                batch_labels.append(t)
                batch_x.append(self.dataset.get_item(t, s))
            yield torch.cat(batch_x, dim=0), torch.tensor(batch_labels, dtype=torch.float32)


class ODataLoader:
    """
    Generates the dataloader to generate batches of ([x1, x2], [y1, y2]), where x1 and x2 are the images, and
    y1 and y2 are labels of the form s{s}t{t} (eg. s1t1) which correspond to each x. Let t1 and t2 correspond to
    x1 and x2, respectively. The ordering of the stacked output will be such that t1 < t2. Will make sure that the
    images are within n_nearby t. If n_nearby is set to zero, this restriction is removed.
    """

    class Config(BaseModel):
        data_dir: str = os.path.join('data', 'lettuce')
        batch_size: int = 16
        shuffle: bool = True
        n_nearby: int = 5       # if set to 0, will uniformly from entire sequence instead of from the n_nearby ones.
    default_config = Config().dict()

    def __init__(self, mode: str, config: dict):
        _ = self.Config(**config).dict()

        # attributes
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.n_nearby = config['n_nearby']

        # init dataset
        dataset = ImageDataset(self.data_dir, mode)
        self.dataset = dataset

        # generate list of s, t pairs.
        self.st_list = []
        for s in dataset.s_list:
            for t in self.dataset.s2t_list[s]:
                self.st_list.append([s, t])
        self.st_list = np.array(self.st_list)

    def __iter__(self):
        """
        Iterate through batches of data samples.

        Generates the batches according to the description in the Class docstring.

        Yields:
            tuple: A tuple containing two elements:
                - batch (list): A list of image samples (x) in the batch.
                - batch_labels (list): A list of labels (eg. s1t3) associated with the data samples: for debugging.
        """

        s_list = self.dataset.s_list
        # generate list of format [[s, t1, t2]] where t1 and t2 are times in s, t1 also always < t2

        # t2 is uniformly sampled, not including t1
        if self.n_nearby == 0:
            st1t2_list = []
            for s in s_list:
                t_list = deepcopy(self.dataset.s2t_list[s])
                np.random.shuffle(t_list)
                st1t2_list.extend([[s, min(ta, tb), max(ta, tb)] for ta, tb in zip(t_list, t_list[1:])])

        # t2 is sampled to be within n_nearby of t1
        else:
            st1t2_list = []
            for s in s_list:
                for t1 in self.dataset.s2t_list[s]:
                    if t1 != max(self.dataset.s2t_list[s]):
                        try:
                            t2 = random.choice([t for t in self.dataset.s2t_list[s] if (t1 < t < t1 + self.n_nearby)])
                            st1t2_list.append([s, t1, t2])
                        except IndexError:
                            print(f'n_nearby = {self.n_nearby} is too strict... for s{s} t{t1}')

        if self.shuffle:
            np.random.shuffle(st1t2_list)

        for start_idx in range(0, len(st1t2_list), self.batch_size//2):
            end_idx = min(start_idx + self.batch_size//2, len(st1t2_list))
            st1t2_batch = st1t2_list[start_idx:end_idx]

            batch_x1, batch_x2 = [], []
            batch_label1, batch_label2 = [], []
            for s, t1, t2 in st1t2_batch:
                batch_x1.append(self.dataset.get_item(t1, s))
                batch_x2.append(self.dataset.get_item(t2, s))
                batch_label1.append(f's{s}t{t1}')
                batch_label2.append(f's{s}t{t2}')

            # stack it, [batch_x1, batch_x2], use labels for sanity checks and debugging
            batch_x = torch.cat([torch.cat(batch_x1, dim=0), torch.cat(batch_x2, dim=0)], dim=0)
            batch_label = batch_label1.copy()
            batch_label.extend(batch_label2)
            yield batch_x, batch_label


class DRDataLoader:
    """
    Dataloader in the format required by classical dimensionality reduction methods - a long list of vectors with
    identifiers for each element.
    """

    class Config(BaseModel):
        data_dir: str = os.path.join('data', 'lettuce')
        mode: str = 'all'
    default_config = Config().dict()

    def __init__(self, mode: str, config: dict):
        _ = self.Config(**config).dict()

        # attribute
        self.data_dir = config['data_dir']

        mode2modes = {
            'all': ['train', 'val', 'test'],
            'train': ['train'],
            'val': ['val'],
            'test': ['test']
        }

        if mode in mode2modes:
            modes = mode2modes[mode]
        else:
            raise NotImplementedError(f'mode ({mode}) not in {mode2modes.keys()}')

        # generate big list of x, and other lists used for identification.
        x_list, t_list, s_list, mode_list = [], [], [], []
        for mode in modes:
            dataset = ImageDataset(self.data_dir, mode)
            with tqdm(total=len(dataset.s_list), desc=f'Loading {mode} data') as pbar:
                for s in dataset.s_list:
                    for t in range(dataset.s2nt[str(s)]):
                        x = np.array(dataset.get_item(t, s))
                        x_list.append(x.reshape(-1))
                        t_list.append(t)
                        s_list.append(s)
                        mode_list.append(mode)
                    pbar.update(1)

        # format
        x_list = np.array(x_list, dtype=np.float32)
        t_list = np.array(t_list)
        s_list = np.array(s_list)

        # normalise
        x_list -= x_list.mean(axis=0)
        x_list /= x_list.std(axis=0)

        # register
        self.x_list = x_list
        self.t_list = t_list
        self.s_list = s_list
        self.mode_list = mode_list

    def get_data(self):
        """
        Returns the big list of images (x) and it's label (t), (s) and (mode)
        """
        return self.x_list, self.t_list, self.s_list, self.mode_list


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

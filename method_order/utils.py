import os
import random
from copy import deepcopy
import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from general.utils import set_seed, ImageDataset
from method_order.models import CNN
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
    dataloader = ODataLoader(mode, config)
    return dataloader


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

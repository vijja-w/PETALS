import os
import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from general.utils import set_seed, ImageDataset
from method_t.models import CNN
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
    dataloader = TDataLoader(mode, config)
    return dataloader


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
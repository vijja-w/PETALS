import os
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from tqdm import tqdm

from general.utils import set_seed, ImageDataset
from method_dimred.models import ModelTSNE, ModelPCA, ModelUMAP
from pydantic import BaseModel


def model_init(config):
    model_name = config['model_name']
    set_seed(config['seed'])
    if model_name == 'tsne':
        model = ModelTSNE(config)
    elif model_name == 'pca':
        model = ModelPCA(config)
    elif model_name == 'umap':
        model = ModelUMAP(config)
    else:
        raise NotImplementedError(f'model_name ({model_name}) not implemented')
    return model


def data_init(mode, config):
    dataloader = DRDataLoader(mode, config)
    return dataloader


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
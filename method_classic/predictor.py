import os

import torch
torch.set_float32_matmul_precision('high')

from pydantic import BaseModel
from typing import Tuple

import numpy as np
from method_classic.utils import model_init, data_init
from general.utils import (
    normalise_data_dict,
)
from collections import defaultdict
import pickle


class Predictor:
    class Config(BaseModel):
        # model hyper-parameters
        lower_threshold: Tuple[float, float, float] = (0.1159, 0.4194, 0.4196)
        upper_threshold: Tuple[float, float, float] = (0.1935, 1.0000, 0.8706)
        model_name: str = 'colour_threshold'

        # data hyper-parameters
        data_dir: str = os.path.join('data', 'lettuce')
        batch_size: int = 64
        shuffle: bool = True

        # experiment hyper-parameters
        seed: int = 1

    default_config = Config().dict()

    def __init__(self, config):

        self.config = self.Config(**config).dict()
        self.model = model_init(config)

    def train(self, images):
        self.model.train(images)

    def test(self):
        """
        Use the tuned thresholds to generate the embeddings (z) for the plant id (s) and wall-clock age (t). Save this
        as zst.pkl for evaluation later.
        """

        lt = self.model.lower_threshold
        ut = self.model.upper_threshold
        expt_name = f'l_h{lt[0]:.4f}_s{lt[1]:.4f}_v{lt[2]:.4f}_uh{ut[0]:.4f}_s{ut[1]:.4f}_v{ut[2]:.4f}'
        folder_name = f'logs/colour_threshold/{expt_name}'
        file_path = os.path.join(folder_name, f'zst.pkl')
        if os.path.exists(file_path):
            print(f'The file {file_path}, exists. Doing nothing...')
        else:
            zst_out = {}
            for mode in ['train', 'val', 'test']:
                print(f'running inference on {mode} dataset...')
                dataloader = data_init(mode, self.config)
                dataset = dataloader.dataset
                data_dict = self.run_inference(dataset)
                normalised_data_dict = normalise_data_dict(data_dict)  # z = normalised_data_dict[str(s)][t]
                result = defaultdict(list)
                zst = []
                for s, z_t in normalised_data_dict.items():
                    for t, z in enumerate(z_t):
                        result[t].append(z)
                        zst.append([z, s, t])
                zst_out[mode] = zst
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(file_path, 'wb') as file:
                pickle.dump(zst_out, file)

    def run_inference(self, dataset):
        """
        Runs the model on the given dataset.

        Args:
            dataset (ImageDataset): The dataset that will be used to run the model on.

        Returns:
            data (dict): results from the model. of the format data[s] = z, where len(z) = nt for that s.
        """

        data = {}
        for i, s in enumerate(dataset.s_list):
            data[s] = []
            nt = dataset.s2nt[f'{s}']
            for t in range(nt):
                image = dataset.get_item(t, s)
                z = self.model(image)
                data[s].append(z.reshape(1, -1))
            data[s] = np.concatenate(data[s], axis=0)
        return data


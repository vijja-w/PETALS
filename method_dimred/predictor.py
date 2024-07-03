import os
import pickle

import torch
torch.set_float32_matmul_precision('high')

from pydantic import BaseModel

from method_dimred.utils import model_init, data_init
from general.utils import get_run, set_seed


class Predictor:
    class Config(BaseModel):

        # model hyper-parameters
        latent_dim: int = 5
        seed: int = 42

        model_name: str = 'umap'
        n_neighbours: int = 200
        t_supervised: bool = True

        # data hyper-parameters
        data_dir: str = os.path.join('data', 'lettuce')

    default_config = Config().dict()

    def __init__(self, config):

        self.config = self.Config(**config).dict()
        self.model = model_init(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, expt_name):
        """
        Trains the machine learning model.

        Args:
            expt_name (str): the name of the experiment to load.
            project_name (str): the project_name the experiment belongs to.
        """
        set_seed(self.config['seed'])
        folder_name = f'logs/dimred/{expt_name}'
        file_path = os.path.join(folder_name, f'zst.pkl')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if os.path.exists(file_path):
            print(f'The file {file_path}, exists. Doing nothing...')
        else:
            if self.config['model_name'] == 'tsne':
                dataloader = data_init('all', self.config)
            else:
                dataloader = data_init('train', self.config)
            x_list, t_list, s_list, mode_list = dataloader.get_data()

            print(f"Fitting model: {self.config['model_name']}")
            if self.config['model_name'] == 'tsne':
                z_list = self.model.train(x_list, t_list)
            else:
                self.model.train(x_list, t_list)
                dataloader = data_init('all', self.config)
                x_list, t_list, s_list, mode_list = dataloader.get_data()
                z_list = self.model.test(x_list)

            zst_out = {'train': [], 'val': [], 'test': []}
            for z, s, t, mode in zip(z_list, s_list, t_list, mode_list):
                zst_out[mode].append([z, s, t])

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(file_path, 'wb') as file:
                pickle.dump(zst_out, file)

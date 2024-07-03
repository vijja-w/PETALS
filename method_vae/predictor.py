import os
import wandb

import torch
torch.set_float32_matmul_precision('high')

from pydantic import BaseModel
from typing import Tuple

import numpy as np
from method_vae.utils import model_init, data_init
from general.utils import get_run, set_seed
from general.utils import (
    normalise_data_dict,
    make_same_length,
    plot2d_interactive,
    plot2d,
    plot_latent_map,
    run_tsne
)


class Predictor:
    class Config(BaseModel):
        # model hyper-parameters
        input_channels: int = 3
        latent_dim: int = 2
        encoder_conv_dims: Tuple[int, ...] = (64, 128, 256, 512)
        encoder_fc_dims: Tuple[int, ...] = (1024, 1024)
        learning_rate: float = 1e-6
        beta: float = 1.0
        model_name: str = 'cnn'
        seed: int = 1

        # data hyper-parameters
        data_dir: str = os.path.join('data', 'lettuce')
        batch_size: int = 16
        shuffle: bool = True

        # experiment hyper-parameters
        max_epoch: int = 500
        patience: int = 50

    default_config = Config().dict()

    def __init__(self, config):
        self.config = self.Config(**config).dict()
        self.model = model_init(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, expt_name, project_name):
        """
        Trains the machine learning model.

        Args:
            expt_name (str): the name of the experiment to load.
            project_name (str): the project_name the experiment belongs to.
        """
        set_seed(self.config['seed'])

        run = wandb.init(project=project_name, config=self.config, name=expt_name)
        wandb.watch(self.model, log_freq=5)

        max_epoch = self.config['max_epoch']
        patience = self.config['patience']

        train_dataloader = data_init('train', self.config)
        val_dataloader = data_init('val', self.config)

        self.model.to(self.device)
        optimizer = self.model.configure_optimizers()
        best_valid_loss = float('inf')
        current_patience = 0

        for epoch in range(max_epoch):
            wandb.log({"epoch": epoch})

            # train
            self.model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                loss = self.model.training_step(batch, batch_idx)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            valid_loss = 0.0
            total = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    loss = self.model.validation_step(batch, batch_idx)
                    valid_loss += loss.item()
                    total += batch[0].shape[0]
            wandb.log({"val_loss": valid_loss})

            # Check for early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                current_patience = 0
                best_model_state = self.model.state_dict()
            else:
                current_patience += 1
            if current_patience >= patience:
                break

        temp_model_path = 'model.pth'
        wandb.log({'best_val_loss': best_valid_loss})
        torch.save(best_model_state, temp_model_path)
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(temp_model_path)
        run.log_artifact(artifact)
        wandb.finish()
        os.remove(temp_model_path)

    def test(self, expt_name, project_name):
        """
        Tests the machine learning model, generating inference plots on the training, validation and test set.
        This is sent to the wandb repository.

        Args:
            expt_name (str): the name of the experiment to load.
            project_name (str): the project_name the experiment belongs to.
        """

        run_id = [run.id for run in wandb.Api().runs(project_name) if run.name == expt_name][0]
        wandb.init(project=project_name, id=run_id, resume='must')

        idx = 0
        for mode in ['test', 'val', 'train']:
            dataset = data_init(mode, self.config).dataset
            print(f'Running inference on: {mode}')
            data_dict = self.run_inference(dataset)
            normalised_data_dict = normalise_data_dict(data_dict)

            if self.model.latent_dim > 2:
                normalised_data_dict = normalise_data_dict(run_tsne(normalised_data_dict))

            # Latent Space Map
            table = wandb.Table(columns=[f'Visualisations - {mode}'])
            size = 0.1
            fig = plot_latent_map(normalised_data_dict, size, self.config['data_dir'], title=f'Map - {mode}')
            table.add_data(wandb.Html(fig.to_html()))
            wandb.log({f'{idx}': table})
            idx += 1

            # Trajectory Overview
            table = wandb.Table(columns=[f'Visualisations - {mode}'])
            fig = plot2d(normalised_data_dict, title=f'Trajectories - {mode}')
            table.add_data(wandb.Html(fig.to_html()))
            wandb.log({f'{idx}': table})
            idx += 1

            # Interactive Trajectory Plot
            table = wandb.Table(columns=[f'Visualisations - {mode}'])
            fig = plot2d_interactive(make_same_length(normalised_data_dict), title=f'Evolution - {mode}')
            table.add_data(wandb.Html(fig.to_html()))
            wandb.log({f'{idx}': table})
            idx += 1
        wandb.finish()

    def run_inference(self, dataset):
        """
        Runs the model on the given dataset.

        Args:
            dataset (ImageDataset): The dataset that will be used to run the model on.

        Returns:
            data (dict): results from the model. of the format data[s] = z, where len(z) = nt for that s.
        """

        self.model.to(self.device)
        self.model.eval()
        data = {}

        for i, s in enumerate(dataset.s_list):
            data[s] = []
            nt = dataset.s2nt[f'{s}']
            for t in range(nt):
                image = dataset.get_item(t, s).to(self.device)
                z, _, _ = self.model.encode(image)
                z = z.detach().cpu().numpy()
                data[s].append(z.reshape(1, -1))
            data[s] = np.concatenate(data[s], axis=0)
        return data

    @classmethod
    def load(cls, expt_name, project_name, data_dir=None):
        """
        Loads a trained model the wandb repository, given the experiment name and project name.

        Args:
            expt_name (str): The name of the experiment to load.
            project_name (str): The project name for the wandb repository.
            data_dir (str): The directory of the data to use. Used during testing.

        Returns:
            Predictor: An instance of the Predictor class with the loaded model.

        Raises:
            ValueError: If the specified experiment could not be found in the project repository.
        """

        run = get_run(expt_name, project_name)
        config = run.config

        pth = [a for a in run.logged_artifacts() if a.type == 'model'][0]
        pth_dir = pth.download()
        pth_file = os.path.join(pth_dir, 'model.pth')
        if data_dir is not None:
            config['data_dir'] = data_dir

        predictor = cls(config)
        checkpoint = torch.load(pth_file)
        predictor.model.load_state_dict(checkpoint)
        return predictor

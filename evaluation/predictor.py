import os
import pickle
import random
import wandb

import torch
torch.set_float32_matmul_precision('high')

from pydantic import BaseModel

import numpy as np
from evaluation.utils import (
    model_init,
    data_init,
    get_tau2st,
    normalise_dim_red,
    plot_2D_latent_assignment,
    plot_image_assignment
)
from general.utils import get_run, set_seed



class Predictor:
    class Config(BaseModel):
        # model hyper-parameter
        learning_rate: float = 1e-2
        model_name: str = 'mlp_small'
        seed: int = 1

        # dataset hyper-parameter
        batch_size: int = 32
        shuffle: bool = True

        # run hyper-parameter
        max_epoch: int = 500
        patience: int = 50
        n_run: int = 5
        dim_red_method: str = 'tsne'
        data_dir: str = os.path.join('data', 'lettuce')

        # source experiment
        source_project_name: str = 'PETALS'
        source_expt_name: str = 't_1'
        method: str = 't'

    default_config = Config().dict()

    def __init__(self, config):

        self.config = self.Config(**config).dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, expt_name, project_name):
        """
        Trains the machine learning model.

        Args:

        """
        set_seed(self.config['seed'])

        n_run = self.config['n_run']
        max_epoch = self.config['max_epoch']
        patience = self.config['patience']

        train_dataloader = data_init('train', self.config)
        val_dataloader = data_init('val', self.config)
        test_dataloader = data_init('test', self.config)
        self.config['input_dim'] = train_dataloader.z.shape[1]
        run = wandb.init(project=project_name, config=self.config, name=expt_name)

        test_losses = []
        for nr, new_seed in enumerate(random.sample(range(0, 1000), n_run)):
            set_seed(new_seed)
            self.model = model_init(self.config)
            self.model.to(self.device)
            optimizer = self.model.configure_optimizers()

            best_valid_loss = float('inf')
            current_patience = 0
            for epoch in range(max_epoch):
                wandb.log({f"epoch_{nr}": epoch})

                # train
                self.model.train()
                for batch_idx, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    loss = self.model.training_step(batch, batch_idx)
                    wandb.log({f'train_loss_{nr}': loss})
                    loss.backward()
                    optimizer.step()

                # validate
                self.model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        loss = self.model.validation_step(batch, batch_idx)
                        valid_loss += loss.item()
                valid_loss /= val_dataloader.z.shape[0]
                wandb.log({f"mean_val_loss_{nr}": valid_loss})

                # Check for early stopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    current_patience = 0
                    best_model_state = self.model.state_dict()
                else:
                    current_patience += 1
                if current_patience >= patience:
                    break
            wandb.log({f'best_mean_val_loss_{nr}': best_valid_loss})

            # test
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    loss = self.model.validation_step(batch, batch_idx)
                    test_loss += loss.item()
            test_loss /= test_dataloader.z.shape[0]
            test_losses.append(test_loss)

            # save and cleanup
            temp_model_path = f'model_{nr}.pth'
            torch.save(best_model_state, temp_model_path)
            artifact = wandb.Artifact(f'model_{nr}', type='model')
            artifact.add_file(temp_model_path)
            run.log_artifact(artifact)
            del best_model_state
            os.remove(temp_model_path)

        for nr_, test_loss in enumerate(test_losses):
            wandb.log({f'test_loss_{nr_}': test_loss})
        wandb.log({'test_loss_mean': np.mean(test_losses)})
        wandb.log({'test_loss_std': np.std(test_losses)})
        wandb.log({'best_nr': test_losses.index(min(test_losses))})
        wandb.finish()

    def assign(self, save=True, load_best=True, test_only=False):
        """


        Args:


        Returns:

        """

        folder_name = f"logs/{self.config['method']}/{self.config['source_expt_name']}"
        file_path = os.path.join(folder_name, f'tauzst.pkl')
        if load_best:
            if os.path.exists(file_path):
                print('pre-computed tauzst found, loading...')
                with open(file_path, 'rb') as file:
                    tauzst = pickle.load(file)
                return tauzst

        if test_only:
            modes = ['test']
        else:
            modes = ['train', 'val', 'test']
        config = self.config
        config['shuffle'] = False
        dataloaders = {mode: data_init(mode, config) for mode in modes}

        self.model.to(self.device)
        self.model.eval()
        tauzst = {mode: {'tau': [], 'z': [], 's': [], 't': []} for mode in modes}
        for mode in modes:
            for i, (z, s, t) in enumerate(dataloaders[mode]):
                tau = self.model(z.to(self.device)).detach().cpu().numpy()
                tauzst[mode]['tau'].append(tau)
                tauzst[mode]['z'].append(z)
                tauzst[mode]['s'].append(s)
                tauzst[mode]['t'].append(t)
            tauzst[mode]['tau'] = np.concatenate(tauzst[mode]['tau']).reshape(-1)
            tauzst[mode]['z'] = np.concatenate(tauzst[mode]['z'])
            tauzst[mode]['s'] = np.concatenate(tauzst[mode]['s']).reshape(-1)
            tauzst[mode]['t'] = np.concatenate(tauzst[mode]['t']).reshape(-1)

        if save:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(file_path, 'wb') as file:
                pickle.dump(tauzst, file)

        return tauzst

    def test(self, expt_name, project_name):
        """
        Tests the machine learning model, generating inference plots on the training, validation and test set.
        This is sent to the wandb repository.
        """

        run_id = [run.id for run in wandb.Api().runs(project_name) if run.name == expt_name][0]
        wandb.init(project=project_name, id=run_id, resume='must')

        tauzst = self.assign()
        tau2st = get_tau2st(tauzst)
        tauzst, x_lims, y_lims = normalise_dim_red(tauzst, self.config)
        for mode in ['test', 'val', 'train']:
            # plot 2D latent assignment
            print(f'Generating latent assignment plots on the {mode} dataset')
            _ = plot_2D_latent_assignment(tauzst, mode, x_lims, y_lims, self.config)

            # plot image assignment
            print(f'Generating image assignment plots on the {mode} dataset')
            _ = plot_image_assignment(tau2st, mode, self.config, save=True)
        wandb.finish()

    @classmethod
    def load(cls, expt_name, project_name, data_dir=None, nr=None):
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
        if data_dir is not None:
            config['data_dir'] = data_dir

        if nr is None:
            history = run.history(run.lastHistoryStep)
            nr = int(history['best_nr'].unique()[1])
        pth_list = [a for a in run.logged_artifacts() if f'model_{nr}' in a.name]
        if len(pth_list) == 0:
            raise ValueError(f'nr ({nr}) not found.')
        pth = pth_list[0]
        pth_dir = pth.download()
        pth_file = os.path.join(pth_dir, f'model_{nr}.pth')
        checkpoint = torch.load(pth_file)
        predictor = cls(config)
        predictor.model = model_init(config)
        predictor.model.load_state_dict(checkpoint)
        return predictor


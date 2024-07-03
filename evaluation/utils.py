from general.utils import normalise_data_dict, set_seed
from evaluation.models import MLPSmall
from collections import defaultdict
import pickle
import os
import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from pydantic import BaseModel

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.subplots import make_subplots
from plotly.offline import plot

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy

from general.utils import ImageThumbnail
from tqdm import tqdm

import wandb


def model_init(config):
    model_name = config['model_name']
    if model_name == 'mlp_small':
        model = MLPSmall(config)
    else:
        raise NotImplementedError(f'model_name ({model_name}) not implemented')
    return model


def data_init(mode, config):
    dataloader = EDataLoader(mode, config)
    return dataloader


def get_zst(source_expt_name, source_project_name, method, save=True):
    allowed_methods = ['contrast', 't', 'order', 'colour_threshold', 'dimred', 'vae']
    if method not in allowed_methods:
        raise ValueError(f'The specified method ({method}) is not in the {allowed_methods}')

    folder_name = f'logs/{method}/{source_expt_name}'
    file_path = os.path.join(folder_name, f'zst.pkl')
    if os.path.exists(file_path):
        print('pre-computed zst found, loading...')
        with open(file_path, 'rb') as file:
            zst_out = pickle.load(file)
        return zst_out

    print('pre-computed zst not found.')
    if method == 'contrast':
        from method_contrast.predictor import Predictor
        from method_contrast.utils import CDataLoader as dataloader

    elif method == 't':
        from method_t.predictor import Predictor
        from method_t.utils import TDataLoader as dataloader

    elif method == 'order':
        from method_order.predictor import Predictor
        from method_order.utils import ODataLoader as dataloader

    elif method == 'vae':
        from method_vae.predictor import Predictor
        from method_vae.utils import VDataLoader as dataloader

    else:
        raise ValueError(f'The specified method ({method}) is not in the {allowed_methods}')

    predictor = Predictor.load(source_expt_name, source_project_name)
    config = predictor.config

    zst_out = {}
    for mode in ['train', 'val', 'test']:
        print(f'running inference on {mode} dataset...')
        dataset = dataloader(mode, config).dataset
        data_dict = predictor.run_inference(dataset)

        result = defaultdict(list)
        zst = []
        for s, z_t in data_dict.items():
            for t, z in enumerate(z_t):
                result[t].append(z)
                zst.append([z, s, t])
        zst_out[mode] = zst
    if save:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(file_path, 'wb') as file:
            pickle.dump(zst_out, file)
    return zst_out


class EDataLoader:
    class Config(BaseModel):
        mode: str = 'train'
        batch_size: int = 32
        shuffle: bool = True
        source_expt_name: str = 't_1'
        source_project_name: str = 'PETALS'
        method: str = 't'

    default_config = Config().dict()

    def __init__(self, mode: str, config: dict):
        allowed_modes = ['train', 'val', 'test']
        if mode not in allowed_modes:
            raise ValueError(f'mode ({mode}) not in allowed_modes ({allowed_modes})')

        _ = self.Config(**config).dict()

        self.mode = mode
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']

        self.source_expt_name = config['source_expt_name']
        self.source_project_name = config['source_project_name']
        self.method = config['method']

        zst = get_zst(self.source_expt_name, self.source_project_name, self.method)
        self.z = torch.tensor([list(z) for z, _, _ in zst[mode]], dtype=torch.float32)
        self.s = torch.tensor([[s] for _, s, _ in zst[mode]], dtype=torch.float32)
        self.t = torch.tensor([[t] for _, _, t in zst[mode]], dtype=torch.float32)

    def __iter__(self):
        """
        Iterate through batches of data samples.

        Generates the batches according to the description in the Class docstring.

        Yields:
            tuple: A tuple containing two elements:
                - batch (list): A list of data samples in the batch.
                - batch_labels (list): A list of labels associated with the data samples in the batch for debugging.

        """

        indices = np.arange(len(self.z))
        if self.shuffle and self.mode not in ['val', 'test']:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            yield self.z[batch_indices], self.s[batch_indices], self.t[batch_indices]


def plot_2D_latent_assignment(tauzst, mode, x_lims, y_lims, config, save=True):
    tauzst_ = tauzst[mode]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Assigned tau", "Actual t"))

    cmin = min(
        min(np.concatenate([tauzst_['tau'] for tauzst_ in tauzst.values()])),
        min(np.concatenate([tauzst_['t'] for tauzst_ in tauzst.values()]))
    )
    cmax = max(
        max(np.concatenate([tauzst_['tau'] for tauzst_ in tauzst.values()])),
        max(np.concatenate([tauzst_['t'] for tauzst_ in tauzst.values()]))
    )

    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]
    xmin = x_lims[0] - 0.1*x_range
    xmax = x_lims[1] + 0.1*x_range
    ymin = y_lims[0] - 0.1 * y_range
    ymax = y_lims[1] + 0.1 * y_range

    tau_plot = go.Scatter(
        x=tauzst_['z'][:, 0],
        y=tauzst_['z'][:, 1],
        mode='markers',
        text=tauzst_['tau'],
        marker=dict(
            size=12,
            color=tauzst_['tau'],
            colorscale='Viridis',
            colorbar=dict(title='Tau'),
            cmin=cmin,
            cmax=cmax
        ),
    )
    fig.add_trace(tau_plot, row=1, col=1)

    t_plot = go.Scatter(
        x=tauzst_['z'][:, 0],
        y=tauzst_['z'][:, 1],
        mode='markers',
        text=tauzst_['t'],
        marker=dict(
            size=12,
            color=tauzst_['t'],
            colorscale='Viridis',
            cmin=cmin,
            cmax=cmax
        )
    )
    fig.add_trace(t_plot, row=1, col=2)
    fig.update_layout(title_text=f'2D Latent Assignment: {mode}', showlegend=False)

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    fig.update_xaxes(range=[xmin, xmax], row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], row=1, col=1)
    fig.update_xaxes(range=[xmin, xmax], row=1, col=2)
    fig.update_yaxes(range=[ymin, ymax], row=1, col=2)

    if save:
        folder_name = f"logs/{config['method']}/{config['source_expt_name']}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, f'latent_assignment_{mode}.html')
        plot(fig, filename=file_path, auto_open=False)
        name = f'2D Latent Assignment - {mode}'
        table = wandb.Table(columns=[name])
        table.add_data(wandb.Html(fig.to_html()))
        wandb.log({name: table})
    return fig


def normalise_dim_red(tauzst, config):
    allowed_dim_red_methods = ['tsne', 'pca']
    dim_red_method = config['dim_red_method']
    if dim_red_method not in allowed_dim_red_methods:
        raise NotImplementedError(f'dim_red_method ({dim_red_method}) not in {allowed_dim_red_methods}')

    tauzst_out = copy.deepcopy(tauzst)
    z_all = np.concatenate([tauzst_['z'] for tauzst_ in tauzst_out.values()])
    mean = z_all.mean(axis=0)
    std = z_all.std(axis=0)
    std[std == 0] = 1.0
    z_all = (z_all - mean) / std

    if z_all.shape[1] > 2:
        if dim_red_method == 'tsne':
            dim_reducer = TSNE(n_components=2, perplexity=30, learning_rate=300)
        elif dim_red_method == 'pca':
            dim_reducer = PCA(2)
        else:
            raise NotImplementedError(f'method ({dim_red_method}) not in {allowed_dim_red_methods}')
        z_all = dim_reducer.fit_transform(z_all)
        mean = z_all.mean(axis=0)
        std = z_all.std(axis=0)
        std[std == 0] = 1.0
        z_all = (z_all - mean) / std

    last_index = 0
    for key in tauzst_out.keys():
        new_index = last_index + len(tauzst_out[key]['z'])
        tauzst_out[key]['z'] = z_all[last_index:new_index]
        last_index = new_index

    x_lims = (min(z_all[:, 0]), max(z_all[:, 0]))
    y_lims = (min(z_all[:, 1]), max(z_all[:, 1]))
    return tauzst_out, x_lims, y_lims


def get_tau2st(tauzst):
    tau2st = {}
    for mode in ['train', 'val', 'test']:
        tau_array = np.round(tauzst[mode]['tau']).astype(int)
        s_array = tauzst[mode]['s'].astype(int)
        t_array = tauzst[mode]['t'].astype(int)
        tau2st_ = defaultdict(list)
        for tau, s, t in zip(tau_array, s_array, t_array):
            tau2st_[tau].append((s, t))
        tau2st[mode] = tau2st_
    return tau2st


def plot_image_assignment(tau2st, mode, config, size=0.005, limit=50, save=True):
    x_limits = (-2 * size, size * (len(tau2st[mode].keys()) + 2))
    y_range = size * (min(limit, max([len(st_list) for st_list in tau2st[mode].values()])) + 2)
    y_limits = ((-y_range / 2) - size, (y_range / 2) - size)

    fig = go.Figure()
    fig.update_yaxes(visible=False, range=y_limits)
    fig.update_xaxes(visible=False, range=x_limits, scaleanchor="y")

    i = 0
    image_thumbnail = ImageThumbnail(config['data_dir'])
    with tqdm(total=len(tau2st[mode].keys())) as pbar:
        for key in np.sort([key for key in tau2st[mode].keys()]):
            st_list = tau2st[mode][key]
            j = -size * min(limit, len(st_list)) / 2
            for s, t in st_list:
                if j / size > limit / 2:
                    break

                thumbnail = image_thumbnail.get_item(t, s)

                fig.add_layout_image(dict(xref='x', yref='y', x=i, sizex=size, y=j, sizey=size,
                                          layer="above", sizing="stretch", source=thumbnail))
                j += size
            i += size
            pbar.update(1)
    fig.update_layout(title='Image Assignment', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    if save:
        name = f'Image Assignment - {mode}'
        folder_name = f"logs/{config['method']}/{config['source_expt_name']}"
        file_path = os.path.join(folder_name, f'image_assignment_{mode}.png')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        fig.write_image(file_path, scale=3)
        images = wandb.Image(file_path)
        wandb.log({name: images})

    return fig

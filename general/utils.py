import os
import json
import shutil
import random

from tqdm import tqdm
from collections import defaultdict

import wandb
import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_expt_name(expt_name, project_name):
    api = wandb.Api()
    runs = api.runs(f'{api.default_entity}/{project_name}')
    expt_names = [run.name for run in runs]
    if len(expt_names) != 0:
        if expt_name in expt_names:
            raise ValueError(f'expt_name = {expt_name} is already taken')


def get_run(expt_name, project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == expt_name:
            return run
    raise ValueError(f'expt_name = {expt_name} not found.')


def clean_up():
    if os.path.exists('artifacts') and os.path.isdir('artifacts'):
        try:
            shutil.rmtree('artifacts')
        except Exception as e:
            print(f"Clean up error - couldn't delete 'artifacts' directory: {e}")

    files = os.listdir('wandb')
    for file in files:
        file_path = os.path.join('wandb', file)
        if os.path.isdir(file_path):
            try:
                shutil.rmtree(file_path)
            except Exception as e:
                print(f"Clean up error - couldn't delete file {file}: {e}")


def datadict2datastacked(data_dict):
    data_stacked = []
    for s in data_dict:
        data_stacked.append(data_dict[s])
    data_stacked = np.concatenate(data_stacked, axis=0)
    return data_stacked


def normalise_data_dict(data_dict):
    data_stacked = datadict2datastacked(data_dict)
    mean = np.mean(data_stacked, axis=0)
    std_dev = np.std(data_stacked, axis=0)
    std_dev[std_dev == 0] = 1.0 # remove zero std_dev
    for s in data_dict:
        data_dict[s] = (data_dict[s] - mean) / std_dev
    return data_dict


def make_same_length(normalised_data_dict):
    max_len = 0
    for data in normalised_data_dict.values():
        if len(data) > max_len:
            max_len = len(data)

    for s in normalised_data_dict:
        if len(normalised_data_dict[s]) != max_len:
            n_copy = max_len - normalised_data_dict[s].shape[0]
            normalised_data_dict[s] = np.concatenate(
                [normalised_data_dict[s], normalised_data_dict[s][-1].reshape(1, -1).repeat(n_copy, axis=0)],
                axis=0)
    return normalised_data_dict


def plot2d_interactive(data_dict, title='State Evolution'):
    duration = 250
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time Index:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": duration},
        "steps": []
    }
    cmap = plotly.colors.qualitative.Plotly

    # initial line
    for c, s in enumerate(data_dict):
        d = {
            "x": np.array([data_dict[s][0, 0]]),
            "y": np.array([data_dict[s][0, 1]]),
            "mode": "lines",
            "text": f'{s}',
            "name": f'{s}',
            "visible": True,
            "line": {"color": cmap[c % len(cmap)],
                     "width": 1}

        }
        fig_dict["data"].append(d)

    # initial marker
    for c, s in enumerate(data_dict):
        d = {
            "x": np.array([data_dict[s][0, 0]]),
            "y": np.array([data_dict[s][0, 1]]),
            "mode": "markers",
            "text": f'{s}',
            "name": f'{s}',
            "visible": True,
            "marker": {"color": cmap[c % len(cmap)], "size": 5}
        }
        fig_dict["data"].append(d)

    # make frames
    for t in range(len(next(iter(data_dict.values())))):
        frame = {"data": [], "name": str(t)}

        # frame lines
        for c, s in enumerate(data_dict):
            d = {
                "x": data_dict[s][0:t+1, 0],
                "y": data_dict[s][0:t+1, 1],
                "mode": "lines",
                "text": f'{s}',
                "name": f'{s}',
                "line": {"color": cmap[c % len(cmap)],
                         "width": 1}
            }
            frame["data"].append(d)

        # frame markers
        for c, s in enumerate(data_dict):
            d = {
                "x": np.array([data_dict[s][t, 0]]),
                "y": np.array([data_dict[s][t, 1]]),
                "mode": "markers",
                "text": f'{s}',
                "name": f'{s}',
                "marker": {"color": cmap[c % len(cmap)], "size": 5}
            }
            frame["data"].append(d)

        # build steps
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [t],
            {
                "frame": {"duration": duration, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": duration}
            }
        ],
            "label": t,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)

    # set plot axis limits
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for data in data_dict.values():
        minx, miny = data.min(axis=0)
        maxx, maxy = data.max(axis=0)
        if minx < min_x:
            min_x = minx
        if miny < min_y:
            min_y = miny
        if maxx > max_x:
            max_x = maxx
        if maxy > max_y:
            max_y = maxy

    range_x = max_x - min_x
    range_y = max_y - min_y
    x_axis_limits = [min_x - 0.1*range_x, max_x + 0.1*range_x]
    y_axis_limits = [min_y - 0.1*range_y, max_y + 0.1*range_y]

    fig.update_layout(xaxis=dict(range=x_axis_limits), yaxis=dict(range=y_axis_limits), showlegend=False, title=title)
    return fig


def plot2d(data_dict, title='State Evolution'):
    traces = []
    for s in data_dict:
        traces.append(go.Scatter(x=data_dict[s][:, 0], y=data_dict[s][:, 1], mode='lines+markers',
                                 line=dict(width=1),
                                 marker=dict(symbol='circle', size=5, opacity=1),
                                 hovertext=np.arange(len(data_dict[s][:, 0])),
                                 name=f's = {s}'))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        showlegend=True,
    )
    return fig


class ImageThumbnail:
    def __init__(self, data_dir=os.path.join('data', 'will_5')):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)
        self.extension = metadata['extension']
        self.s2nt = metadata['s2nt']
        self.s_list = [s for mode in ['train', 'val', 'test'] for s in metadata[mode]]
        self.s2t_list = defaultdict(list)
        for s in self.s_list:
            for t in sorted([int(f.split('t')[1].split(f'.{self.extension}')[0]) for f in os.listdir(data_dir + f'/s{s}')]):
                self.s2t_list[s].append(t)

    def get_item(self, t, s):
        image_path = os.path.join(self.data_dir, f's{s}', f't{t}.{self.extension}')
        image = Image.open(image_path)
        return image


def plot_latent_map(normalised_data_dict, size, data_dir, title='Latent Map'):
    image_thumbnail = ImageThumbnail(data_dir)

    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for data in normalised_data_dict.values():
        minx, miny = data.min(axis=0)
        maxx, maxy = data.max(axis=0)
        if minx < min_x:
            min_x = minx
        if miny < min_y:
            min_y = miny
        if maxx > max_x:
            max_x = maxx
        if maxy > max_y:
            max_y = maxy

    range_x = max_x - min_x
    range_y = max_y - min_y
    x_axis_limits = [min_x - 0.05 * range_x, max_x + 0.05 * range_x]
    y_axis_limits = [min_y - 0.05 * range_y, max_y + 0.05 * range_y]

    s_filtered, t_idx_filtered, z_filtered = [], [], []
    for s in normalised_data_dict:
        for t_idx in range(len(normalised_data_dict[s])):
            if len(z_filtered) == 0:
                s_filtered.append(s)
                t_idx_filtered.append(t_idx)
                z_filtered.append(normalised_data_dict[s][t_idx])
            else:
                z = normalised_data_dict[s][t_idx]
                distances = np.linalg.norm(np.array(z_filtered) - z, axis=1)
                if np.all(distances > size / np.sqrt(2)):
                    s_filtered.append(s)
                    t_idx_filtered.append(t_idx)
                    z_filtered.append(normalised_data_dict[s][t_idx])
    z_filtered = np.array(z_filtered)

    fig = go.Figure()
    fig.update_xaxes(visible=True, range=x_axis_limits)
    fig.update_yaxes(visible=True, range=y_axis_limits, scaleanchor="x")
    for s, t_idx, z in zip(s_filtered, t_idx_filtered, z_filtered):
        t = image_thumbnail.s2t_list[s][t_idx]
        thumbnail = image_thumbnail.get_item(t, s)
        fig.add_layout_image(dict(xref='x', yref='y', x=z[0], sizex=size, y=z[1], sizey=size,
                                  layer="above", sizing="stretch", source=thumbnail))
    fig.update_layout(title=title)
    return fig


def run_pca(normalised_data_dict, n_components=2):
    pca = PCA(n_components)
    pca.fit(datadict2datastacked(normalised_data_dict))
    reduced_data_dict = {}
    for s in normalised_data_dict:
        reduced_data_dict[s] = pca.transform(normalised_data_dict[s])
    return reduced_data_dict


def run_tsne(normalised_data_dict, n_components=2, perplexity=30, learning_rate=300):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    data_stacked = datadict2datastacked(normalised_data_dict)
    nt_list = [len(normalised_data_dict[s]) for s in normalised_data_dict]

    reduced_data_stacked = tsne.fit_transform(data_stacked)

    reduced_data_dict = {}
    prev_index = 0
    for s, nt in zip(normalised_data_dict, nt_list):
        reduced_data_dict[s] = reduced_data_stacked[prev_index:prev_index + nt]
        prev_index += nt
    return reduced_data_dict


def plot3d_interactive(data_dict, title='State Evolution'):

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time Index:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 10},
        "steps": []
    }
    cmap = plotly.colors.qualitative.Plotly

    # initial line
    for c, s in enumerate(data_dict):
        d = {
            "type": 'scatter3d',
            "x": np.array([data_dict[s][0, 0]]),
            "y": np.array([data_dict[s][0, 1]]),
            "z": np.array([data_dict[s][0, 2]]),
            "mode": "lines",
            "text": f'{s}',
            "name": f'{s}',
            "visible": True,
            "line": {"color": cmap[c % len(cmap)]}

        }
        fig_dict["data"].append(d)

    # initial marker
    for c, s in enumerate(data_dict):
        d = {
            "type": 'scatter3d',
            "x": np.array([data_dict[s][0, 0]]),
            "y": np.array([data_dict[s][0, 1]]),
            "z": np.array([data_dict[s][0, 2]]),
            "mode": "markers",
            "text": f'{s}',
            "name": f'{s}',
            "visible": True,
            "marker": {"color": cmap[c % len(cmap)], "size": 5}
        }
        fig_dict["data"].append(d)

    # make frames
    for t in range(len(next(iter(data_dict.values())))):
        frame = {"data": [], "name": str(t)}

        # frame lines
        for c, s in enumerate(data_dict):
            d = {
                "type": 'scatter3d',
                "x": data_dict[s][0:t+1, 0],
                "y": data_dict[s][0:t+1, 1],
                "z": data_dict[s][0:t+1, 2],
                "mode": "lines",
                "text": f'{s}',
                "name": f'{s}',
                "line": {"color": cmap[c % len(cmap)]}
            }
            frame["data"].append(d)

        # frame markers
        for c, s in enumerate(data_dict):
            d = {
                "type": 'scatter3d',
                "x": np.array([data_dict[s][t, 0]]),
                "y": np.array([data_dict[s][t, 1]]),
                "z": np.array([data_dict[s][t, 2]]),
                "mode": "markers",
                "text": f'{s}',
                "name": f'{s}',
                "marker": {"color": cmap[c % len(cmap)], "size": 5}
            }
            frame["data"].append(d)

        # build steps
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [t],
            {
                "frame": {"duration": 10, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 10}
            }
        ],
            "label": t,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)

    # set plot axis limits
    min_x, max_x, min_y, max_y, min_z, max_z = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf
    for data in data_dict.values():
        minx, miny, minz = data.min(axis=0)
        maxx, maxy, maxz = data.max(axis=0)
        if minx < min_x:
            min_x = minx
        if miny < min_y:
            min_y = miny
        if minz < min_z:
            min_z = minz
        if maxx > max_x:
            max_x = maxx
        if maxy > max_y:
            max_y = maxy
        if maxz > max_z:
            max_z = maxz

    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    x_axis_limits = [min_x - 0.1*range_x, max_x + 0.1*range_x]
    y_axis_limits = [min_y - 0.1*range_y, max_y + 0.1*range_y]
    z_axis_limits = [min_z - 0.1 * range_z, max_z + 0.1 * range_z]

    fig.update_layout(scene=dict(xaxis=dict(range=x_axis_limits),
                                 yaxis=dict(range=y_axis_limits),
                                 zaxis=dict(range=z_axis_limits)),
                      showlegend=False,
                      title=title,
                      scene_aspectmode='cube')
    return fig


def plot3d(data_dict):
    traces = []
    for s in data_dict:
        traces.append(go.Scatter3d(x=data_dict[s][:, 0], y=data_dict[s][:, 1], z=data_dict[s][:, 2], mode='lines+markers',
                                   line=dict(width=1),
                                   marker=dict(symbol='circle', size=5, opacity=1),
                                   text=np.arange(len(data_dict[s][:, 0])),
                                   name=f's = {s}'))

    layout = go.Layout(title='State Evolution',
                       scene=dict(
                           xaxis_title='X-axis',
                           yaxis_title='Y-axis',
                           zaxis_title='Z-axis'),
                       showlegend=True)

    fig = go.Figure(data=traces, layout=layout)
    return fig


def plot2d_ntmap(data_dict, title='State Evolution'):
    c_max = max([len(v) for v in data_dict.values()])

    traces = []
    for s in data_dict:
        colorbar_tickvals = np.linspace(0, 1, num=11)
        c = np.arange(len(data_dict[s]))/c_max
        traces.append(go.Scatter(x=data_dict[s][:, 0], y=data_dict[s][:, 1], mode='markers',
                                 marker=dict(symbol='circle', size=12, opacity=1,
                                             color=c, colorscale='Bluered',
                                             colorbar=dict(title='Days',
                                                           tickvals=np.linspace(0, 1, 11),
                                                           ticktext=['' for val in colorbar_tickvals])),
                                 hovertext=np.arange(len(data_dict[s][:, 0])),
                                 name=f's = {s}'))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        showlegend=False,
    )
    return fig


class ImageDataset(Dataset):
    def __init__(self, data_dir=os.path.join('data', 'will_5'), mode: str = 'train', transform='none'):
        self.data_dir = data_dir
        self.mode = mode

        allowed_transforms = ('none', 'normalise', 'colour_jitter')

        if transform not in allowed_transforms:
            raise ValueError(f'specified transform = {transform} not in {allowed_transforms}')

        if transform == 'none':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        elif transform == 'normalise':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                NormalizePerImage()
            ])

        elif transform == 'colour_jitter':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                NormalizePerImage(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)
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


class NormalizePerImage(transforms.Compose):
    def __init__(self):
        super(NormalizePerImage, self).__init__([])

    def __call__(self, img):
        mean = torch.mean(img, dim=(1, 2)).reshape(-1, 1, 1)
        std = torch.std(img, dim=(1, 2)).reshape(-1, 1, 1)
        std *= 2.0  # tighter spread
        img = (img - mean) / std
        return img


def plot_sorted_images(expt_name, project_name, method, title='Biological Age Map'):
    allowed_methods = ['contrast', 't', 'order']
    if method not in allowed_methods:
        raise ValueError(f'The specified method ({method}) is not in the {allowed_methods}')

    if method == 'contrast':
        from method_contrast.predictor import Predictor
        from method_contrast.utils import ContrastiveDataLoader as dataloader
        config = {
            'data_dir': 'data/will_5',
            'ns_batch': 8,
            'nt_batch': 13,
            'p_pos': 0.2,
            'p_pos_window': 0.125,
            'shuffle': True,
            'transform': 'none',
        }

    elif method == 't':
        from method_t.predictor import Predictor
        from method_t.utils import TDataLoader as dataloader
        config = {
            'data_dir': 'data/will_5',
            'batch_size': 16,
            'shuffle': True,
            'transform': 'none',
        }

    elif method == 'order':
        from method_order.predictor import Predictor
        from method_order.utils import ODataLoader as dataloader
        config = {
            'data_dir': 'data/will_5',
            'batch_size': 16,
            'shuffle': True,
            'transform': 'none',
            'mode': 'nearby'
        }
    else:
        raise ValueError(f'The specified method ({method}) is not in the {allowed_methods}')

    predictor = Predictor.load(expt_name, project_name)
    dataset = dataloader('train', config).dataset
    data_dict = predictor.run_inference(dataset)
    normalised_data_dict = normalise_data_dict(data_dict)  # z = normalised_data_dict[str(s)][t]

    # get the average coordinates along the trajectory:
    result = defaultdict(list)
    z_st = []
    for s, z_t in normalised_data_dict.items():
        for t, z in enumerate(z_t):
            result[t].append(z)
            z_st.append([z, s, t])

    result_dict = dict(result)
    avg_zt_list = [np.mean(zs, axis=0) for _, zs in result_dict.items()]

    # assign z to tau
    z_grouped = [[] for _ in range(len(avg_zt_list))]
    for i in range(len(avg_zt_list) - 1):
        new_z_st = []
        for z, s, t in z_st:
            if np.linalg.norm(avg_zt_list[i] - z) < np.linalg.norm(avg_zt_list[i + 1] - z):
                z_grouped[i].append([z, s, t])
            else:
                new_z_st.append([z, s, t])
        z_st = new_z_st
    z_grouped[i + 1] = z_st

    # size = 0.005
    size = 0.00005
    limit = 50

    x_limits = (-2*size, size * (len(z_grouped) + 2))
    y_range = size * (min(limit, max([len(zst_list) for zst_list in z_grouped])) + 2)
    y_limits = ((-y_range / 2) - size, (y_range / 2) - size)

    fig = go.Figure()
    fig.update_yaxes(visible=False, range=y_limits)
    fig.update_xaxes(visible=False, range=x_limits, scaleanchor="y")

    i = 0
    image_thumbnail = ImageThumbnail()
    with tqdm(total=len(z_grouped)) as pbar:
        for zst_list in z_grouped:
            j = -size * min(limit, len(zst_list)) / 2
            for z, s, t in zst_list:
                if j / size > limit / 2:
                    break

                thumbnail = image_thumbnail.get_item(t, s)
                # todo resize here?

                fig.add_layout_image(dict(xref='x', yref='y', x=i, sizex=size, y=j, sizey=size,
                                          layer="above", sizing="stretch", source=thumbnail))
                j += size
            i += size
            pbar.update(1)
    fig.update_layout(title=title, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_sorted_dataset(data_dir, title='Biological Age Map'):

    from method_t.utils import TDataLoader as dataloader
    config = {
        'data_dir': data_dir,
        'batch_size': 16,
        'shuffle': True,
        'transform': 'none',
    }
    dataset = dataloader('train', config).dataset
    image_thumbnail = ImageThumbnail()

    for s in dataset.s_list:
        nt = dataset.s2nt[str(s)]
        for t in range(nt):
            image_thumbnail.get_item(t, s)

    result = defaultdict(list)
    for s in dataset.s_list:
        for t in range(dataset.s2nt[str(s)]):
            result[t].append(s)
    result_dict = dict(result)

    size = 0.005
    limit = 50

    x_limits = (-2 * size, size * (max([dataset.s2nt[str(s)] for s in dataset.s_list]) + 2))
    y_range = size * (len(dataset.s_list) + 2)
    y_limits = ((-y_range / 2) - size, (y_range / 2) - size)

    fig = go.Figure()
    fig.update_yaxes(visible=False, range=y_limits)
    fig.update_xaxes(visible=False, range=x_limits, scaleanchor="y")

    i = 0
    image_thumbnail = ImageThumbnail()
    with tqdm(total=len(result_dict.keys())) as pbar:
        for t in result_dict.keys():
            j = -size * min(limit, len(result_dict[t])) / 2

            for s in result_dict[t]:
                if j / size > limit / 2:
                    break

                thumbnail = image_thumbnail.get_item(t, s)
                # todo resize here?
                original_width, original_height = thumbnail.size
                thumbnail.resize((int(original_width/10), int(original_height/10)), Image.ANTIALIAS)
                fig.add_layout_image(dict(xref='x', yref='y', x=i, sizex=size, y=j, sizey=size,
                                          layer="above", sizing="stretch", source=thumbnail))
                j += size
            i += size
            pbar.update(1)
    fig.update_layout(title=title, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_image_assignment_tau(tau, tau2st, expt_name, size=0.005, limit=50, save=False):
    image_thumbnail = ImageThumbnail()
    st_list = []
    # for mode in ['train', 'val', 'test']:   # todo testing
    for mode in ['train']:      # todo testing
        st_list.extend(tau2st[mode][tau])

    y_limits = (-2 * size, 2 * size)
    x_range = size * (min(limit, len(st_list)))
    x_limits = ((-x_range / 2) - size, (x_range / 2) + size)

    fig = go.Figure()

    fig.update_xaxes(visible=False, range=x_limits)
    fig.update_yaxes(visible=False, range=y_limits, scaleanchor="x")

    i = -size * min(limit, len(st_list)) / 2
    for s, t in st_list:
        if i / size > limit / 2:
            break

        thumbnail = image_thumbnail.get_item(t, s)
        fig.add_layout_image(dict(xref='x', yref='y', x=i, sizex=size, y=0, sizey=size,
                                  layer="above", sizing="stretch", source=thumbnail))
        i += size

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    if save:
        folder_name = f"logs/paper/"
        file_path = os.path.join(folder_name, f'tau{tau}_{expt_name}.png')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        fig.write_image(file_path, scale=3)
    return fig

import torch
torch.set_float32_matmul_precision('high')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from pydantic import BaseModel


class ModelTSNE:
    class Config(BaseModel):
        perplexity: int = 3
        learning_rate: int = 2
        latent_dim: int = 2
        model_name: str = 'tsne'
        seed: int = 42
    default_config = Config().dict()

    def __init__(self, config):
        super().__init__()
        _ = self.Config(**config).dict()

        self.perplexity = config['perplexity']
        self.learning_rate = config['learning_rate']
        self.latent_dim = config['latent_dim']

        self.model_name = config['model_name']
        self.seed = config['seed']

        self.model = TSNE(n_components=self.latent_dim, perplexity=self.perplexity, learning_rate=self.learning_rate)

    def train(self, x, y):
        return self.model.fit_transform(x)


class ModelPCA:
    class Config(BaseModel):
        latent_dim: int = 2
        model_name: str = 'pca'
        seed: int = 42
    default_config = Config().dict()

    def __init__(self, config):
        super().__init__()
        _ = self.Config(**config).dict()
        self.latent_dim = config['latent_dim']
        self.model_name = config['model_name']
        self.seed = config['seed']

        self.model = PCA(n_components=self.latent_dim)

    def train(self, x, y):
        self.model.fit(x)

    def test(self, x):
        return self.model.transform(x)


class ModelUMAP:
    class Config(BaseModel):
        latent_dim: int = 2
        n_neighbours: int = 100
        t_supervised: bool = False
        model_name: str = 'umap'
        seed: int = 42
    default_config = Config().dict()

    def __init__(self, config):
        super().__init__()
        _ = self.Config(**config).dict()
        self.latent_dim = config['latent_dim']
        self.n_neighbours = config['n_neighbours']
        self.t_supervised = config['t_supervised']
        self.model_name = config['model_name']
        self.seed = config['seed']

        self.model = umap.UMAP(n_components=self.latent_dim, n_neighbors=self.n_neighbours)

    def train(self, x, y):
        if self.t_supervised:
            self.model.fit(x, y=y)
        else:
            self.model.fit(x)

    def test(self, x):
        return self.model.transform(x)
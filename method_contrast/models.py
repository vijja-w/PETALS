import torch
from method_contrast.losses import get_loss_simple
torch.set_float32_matmul_precision('high')
import wandb
import torch.nn as nn
from pydantic import BaseModel
from typing import Tuple


class CNN(nn.Module):
    class Config(BaseModel):
        input_channels: int = 3
        latent_dim: int = 2
        feature_dims: Tuple[int, ...] = (64, 128, 256, 512)
        classifier_dims: Tuple[int, ...] = (1024, 1024)
        learning_rate: float = 2e-6
        tau: float = 0.1
        model_name: str = 'cnn'
        seed: int = 1

    default_config = Config().dict()

    def __init__(self, config):
        super().__init__()
        _ = self.Config(**config).dict()

        self.input_dim = config['input_channels']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.model_name = config['model_name']
        self.seed = config['seed']
        self.criterion = ContrastiveLoss(config['tau'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_dims = config['feature_dims']
        self.features = nn.Sequential(
            nn.Conv2d(config['input_channels'], feature_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        for i in range(len(feature_dims) - 1):
            self.features.append(nn.Conv2d(feature_dims[i], feature_dims[i+1], kernel_size=3, padding=1))
            self.features.append(nn.BatchNorm2d(feature_dims[i + 1]))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        classifier_dims = config['classifier_dims']
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dims[-1], classifier_dims[0], kernel_size=7),
            nn.ReLU(inplace=True)
        )

        for i in range(len(classifier_dims) - 1):
            self.classifier.append(nn.Conv2d(classifier_dims[i], classifier_dims[i+1], kernel_size=1))
            self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Conv2d(classifier_dims[-1], config['latent_dim'], kernel_size=1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def factorise_s(self, z, labels):
        # from shape [batch_size, feature_size] -> [ns, nt + 1, feature_size], +1 for anchor
        dim1 = sum([1 for label in labels if label == 0])
        dim2 = int(len(labels) / dim1)
        z = z.reshape(dim1, dim2, -1)
        labels = labels.reshape(dim1, dim2)
        return z, labels

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def training_step(self, batch, batch_idx):
        self.train()
        x, labels_tuple = batch
        labels, _ = labels_tuple
        z = self(x.to(self.device))
        z, labels = self.factorise_s(z, labels)
        loss = self.criterion.get_loss(z, labels.to(self.device))
        wandb.log({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, labels_tuple = batch
        labels, _ = labels_tuple
        z = self(x.to(self.device))
        z, labels = self.factorise_s(z, labels)
        loss = self.criterion.get_loss(z, labels.to(self.device))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class ContrastiveLoss:
    def __init__(self, tau, anchor_index=0):
        self.anchor_index = anchor_index
        self.tau = tau

    def get_loss(self, features, labels):
        return get_loss_simple(features, labels, self.tau)

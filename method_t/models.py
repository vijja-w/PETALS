import torch
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
        self.criterion = nn.MSELoss()
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

        self.output_layers = nn.Sequential(
            nn.Linear(in_features=config['latent_dim'], out_features=10, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        z = self.avgpool(x)
        return z.view(x.size(0), -1)

    def z2output(self, z):
        return self.output_layers(z).view(-1)

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        z = self(x.to(self.device))
        y_pred = self.z2output(z)
        loss = self.criterion(y.to(self.device), y_pred)
        wandb.log({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch
        z = self(x.to(self.device))
        y_pred = self.z2output(z)
        loss = self.criterion(y.to(self.device), y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
import torch
torch.set_float32_matmul_precision('high')
import wandb
import torch.nn as nn
from pydantic import BaseModel
from typing import Tuple


class CNN(nn.Module):
    class Config(BaseModel):
        # model hyper-parameters
        input_channels: int = 3
        feature_dims: Tuple[int, ...] = (64, 128, 256, 512)
        latent_dim: int = 5
        classifier_dims: Tuple[int, ...] = (1024, 1024)
        learning_rate: float = 1e-6
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
        self.criterion = nn.CrossEntropyLoss()
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
            nn.Linear(in_features=config['latent_dim']*2, out_features=10, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=2, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        z = self.avgpool(x)
        return z.view(x.size(0), -1)

    def z2output(self, z):
        split_index = int(len(z)/2)
        z1, z2 = z[0: split_index], z[split_index::]
        z1z2 = torch.cat([z1, z2], dim=1)
        z2z1 = torch.cat([z2, z1], dim=1)

        target_z1z2 = torch.ones(split_index, dtype=torch.long)
        target_z2z1 = torch.zeros(split_index, dtype=torch.long)

        zz = torch.cat([z1z2, z2z1], dim=0)
        target = torch.cat([target_z1z2, target_z2z1], dim=0)

        return self.output_layers(zz), target.to(self.device)

    def training_step(self, batch, batch_idx):
        self.train()
        x, labels = batch
        z = self(x.to(self.device))
        pred, target = self.z2output(z)
        loss = self.criterion(pred, target)
        wandb.log({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, labels = batch
        z = self(x.to(self.device))
        pred, target = self.z2output(z)
        loss = self.criterion(pred, target)

        n_correct_pred = (torch.argmax(pred, dim=1) == target).sum().item()

        return loss, n_correct_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @classmethod
    def check_signature(cls, config):
        """
        Class method to check if the provided configuration dictionary is valid.

        Args:
            config (dict): A dictionary containing configuration parameters for the model.

        Returns:
            bool: True if the configuration is valid; False otherwise.

        Raises:
            ValueError: If the configuration is missing required keys or has incorrect data types.

        Example:
            config = {
                'input_channels': 3,
                'latent_dim': 2
                'feature_dims': (64, 128, 256, 512),
                'classifier_dims': (1024, 1024),
                'learning_rate': 2e-6,
                'model_name': 'cnn',
                'seed': 42
            }
        """

        # Define the required keys and their expected types
        required_keys = [
            ('input_channels', int),
            ('latent_dim', int),
            ('feature_dims', (tuple, list)),
            ('classifier_dims', (tuple, list)),
            ('learning_rate', float),
            ('model_name', str),
            ('seed', int),
        ]

        # Check if all required keys are present and have the correct data types
        for key, expected_type in required_keys:
            if key not in config:
                raise ValueError(f"Missing '{key}' in the configuration.")

            if not isinstance(config[key], expected_type):
                raise ValueError(f"Invalid type for '{key}'. Expected {expected_type}, got {type(config[key])}.")
        return True

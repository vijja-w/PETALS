import torch
torch.set_float32_matmul_precision('high')
import wandb
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple


class CNN(nn.Module):
    class Config(BaseModel):
        input_channels: int = 3
        latent_dim: int = 2
        encoder_conv_dims: Tuple[int, ...] = (64, 128, 256, 512)
        encoder_fc_dims: Tuple[int, ...] = (1024, 1024)
        learning_rate: float = 1e-6
        beta: float = 1.0
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
        self.beta = config['beta']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.expand_dim = 0
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # encoder conv
        encoder_conv_dims = config['encoder_conv_dims']
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(config['input_channels'], encoder_conv_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_conv_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        for i in range(len(encoder_conv_dims) - 1):
            self.encoder_conv.append(nn.Conv2d(encoder_conv_dims[i], encoder_conv_dims[i+1], kernel_size=3, padding=1))
            self.encoder_conv.append(nn.BatchNorm2d(encoder_conv_dims[i + 1]))
            self.encoder_conv.append(nn.ReLU(inplace=True))
            self.encoder_conv.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # encoder fc
        encoder_fc_dims = config['encoder_fc_dims']
        self.encoder_fc = nn.Sequential(
            nn.Conv2d(encoder_conv_dims[-1], encoder_fc_dims[0], kernel_size=7),
            nn.ReLU(inplace=True)
        )
        for i in range(len(encoder_fc_dims) - 1):
            self.encoder_fc.append(nn.Conv2d(encoder_fc_dims[i], encoder_fc_dims[i+1], kernel_size=1))
            self.encoder_fc.append(nn.ReLU(inplace=True))
        self.encoder_fc.append(nn.Conv2d(encoder_fc_dims[-1], 2*config['latent_dim'], kernel_size=1))

        # decoder
        decoder_fc_dims = list(config['encoder_fc_dims'])
        decoder_fc_dims.reverse()
        decoder_conv_dims = list(config['encoder_conv_dims'])
        decoder_conv_dims.reverse()

        # decoder fc
        self.decoder_fc = nn.Sequential(
            nn.ConvTranspose2d(config['latent_dim'], decoder_fc_dims[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        for i in range(len(decoder_fc_dims) - 1):
            self.decoder_fc.append(nn.ConvTranspose2d(decoder_fc_dims[i], decoder_fc_dims[i+1], kernel_size=1))
            self.decoder_fc.append(nn.ReLU(inplace=True))
        self.decoder_fc.append(nn.ConvTranspose2d(decoder_fc_dims[-1], decoder_conv_dims[0], kernel_size=7))

        # decoder conv
        self.decoder_conv = nn.Sequential()
        for i in range(len(encoder_conv_dims) - 1):
            self.decoder_conv.append(nn.ConvTranspose2d(decoder_conv_dims[i], decoder_conv_dims[i+1], kernel_size=3, stride=2))
            self.decoder_conv.append(SliceLayer())
            self.decoder_conv.append(nn.BatchNorm2d(decoder_conv_dims[i + 1]))
            self.decoder_conv.append(nn.ReLU(inplace=True))

            # increase decoder capacity
            self.decoder_conv.append(nn.ConvTranspose2d(decoder_conv_dims[i + 1], decoder_conv_dims[i + 1], kernel_size=3, padding=1))
            self.decoder_conv.append(nn.ReLU(inplace=True))

        self.decoder_conv.append(nn.ConvTranspose2d(decoder_conv_dims[-1], config['input_channels'], kernel_size=3, stride=2))
        self.decoder_conv.append(SliceLayer())
        self.decoder_conv.append(nn.Sigmoid())

    def criterion(self, x_pred, x, mu, logvar):
        recon_loss = F.mse_loss(x_pred.reshape(x_pred.shape[0], -1), x.reshape(x.shape[0], -1), reduction='mean')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kldivergence

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        self.expand_dim = (x.shape[-2], x.shape[-1])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        mu = x[:, 0:self.latent_dim]
        logvar = x[:, self.latent_dim::]
        return self.latent_sample(mu, logvar), mu, logvar

    def decode(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1).expand(z.shape[0], z.shape[1], self.expand_dim[0], self.expand_dim[1]).contiguous()
        x = self.decoder_fc(z)
        x = self.decoder_conv(x)
        return x

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def training_step(self, batch, batch_idx):
        self.train()
        x, labels = batch
        x = x.to(self.device)
        z, mu, logvar = self.encode(x)
        x_pred = self.decode(z)
        loss = self.criterion(x_pred, x, mu, logvar)
        wandb.log({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, labels = batch
        x = x.to(self.device)
        z, mu, logvar = self.encode(x)
        x_pred = self.decode(z)
        loss = self.criterion(x_pred, x, mu, logvar)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class SliceLayer(nn.Module):
    def __init__(self, height_slice=slice(1, None), width_slice=slice(1, None)):
        super().__init__()
        self.height_slice = height_slice
        self.width_slice = width_slice

    def forward(self, x):
        return x[:, :, self.height_slice, self.width_slice]

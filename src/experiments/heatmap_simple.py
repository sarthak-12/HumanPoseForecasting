import argparse
import os
import sys
from os.path import dirname, join

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import wandb
import yaml
from dataset.h36m import Human36M
from models.heatmap_model import HeatmapModel
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn, relu
from torch.nn.init import calculate_gain, xavier_uniform_
from torch.utils.data import DataLoader

wandb.login()

# loading the sweep configuration file
with open(join(dirname(dirname(__file__)), 'heatmap_config.yaml')) as stream:
    config = dict(yaml.safe_load(stream))

train_dataset = Human36M(train=True, heatmap=True)
validation_dataset = Human36M(train=False, heatmap=True)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    drop_last=False,
    num_workers=os.cpu_count(),
)


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(17, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        xavier_uniform_(self.conv1.weight, calculate_gain('relu'))

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        xavier_uniform_(self.conv2.weight, calculate_gain('relu'))

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        xavier_uniform_(self.conv3.weight, calculate_gain('relu'))

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        xavier_uniform_(self.conv4.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = relu(x)

        x = self.conv4(x)

        return x


class SimpleDecoder(nn.Module):
    def __init__(self, hidden_channels: int = 128):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            hidden_channels,
            128,
            kernel_size=3,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.bn1 = nn.BatchNorm2d(128)
        xavier_uniform_(self.conv1.weight, calculate_gain('relu'))

        self.conv2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=3,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.bn2 = nn.BatchNorm2d(64)
        xavier_uniform_(self.conv2.weight, calculate_gain('relu'))

        self.conv3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        xavier_uniform_(self.conv3.weight, calculate_gain('relu'))

        self.conv4 = nn.ConvTranspose2d(32, 17, 3, padding=1)
        xavier_uniform_(self.conv4.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = relu(x)

        x = self.conv4(x)

        return x


def train():
    logger = WandbLogger(save_dir='./wandb/')
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        max_epochs=args.epochs,
        enable_model_summary=False,
    )

    hidden_channels = wandb.config.hidden_channels
    num_layers = wandb.config.num_layers
    dropout = wandb.config.dropout
    lr = wandb.config.lr

    model = HeatmapModel(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )

    # exchanging the encoder and the decoder for a simple 4-layer encoder and decoder respectively
    model.encoder = SimpleEncoder()
    model.decoder = SimpleDecoder(hidden_channels)

    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, validation_loader)
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', '-c', type=int, default=10)
    parser.add_argument('--epochs', '-e', type=int, default=3)
    parser.add_argument('--sweep_name', '-s', type=str)
    args = parser.parse_args()

    if args.sweep_name:
        config['name'] = args.sweep_name

    sweep_id = wandb.sweep(config, project='CudaLab-S22-Project')
    wandb.agent(
        sweep_id,
        function=train,
        project='CudaLab-S22-Project',
        count=args.count,
    )

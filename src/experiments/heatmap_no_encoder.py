import argparse
import os
import sys
from os.path import dirname, join

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import wandb
import yaml
from dataset.h36m import Human36M
from models.conv_gru import ConvGRU
from models.heatmap_model import HeatmapModel
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

wandb.login()

# loading the sweep configuration file
with open(join(dirname(dirname(__file__)), 'heatmap_config.yaml')) as stream:
    config = dict(yaml.safe_load(stream))

train_dataset = Human36M(train=True, heatmap=True)
validation_dataset = Human36M(train=False, heatmap=True)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=4,
    drop_last=False,
    num_workers=os.cpu_count(),
)


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

    # removing the encoder and modifying the recurrent unit to match input channels
    model.encoder = nn.Identity()
    model.rnn = ConvGRU(
        input_channels=17,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )
    # removing the strides from the decoder to keep the output size consistent with the input size
    model.decoder.layer1 = model.decoder._make_layer(
        hidden_channels, 128, stride=1)
    model.decoder.layer2 = model.decoder._make_layer(128, 64, stride=1)

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

import argparse
import os
import sys
from os.path import dirname, join

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import wandb
import yaml
from dataset.h36m import Human36M
from models.conv_lstm import ConvLSTM
from models.heatmap_model import HeatmapModel
from pytorch_lightning.loggers import WandbLogger
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

    # changing the ConvGRU unit to a ConvLSTM unit
    model.rnn = ConvLSTM(
        input_channels=256,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

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

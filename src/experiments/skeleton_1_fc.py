import argparse
import os
import sys
from os.path import dirname, join

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import wandb
import yaml
from dataset.h36m import Human36M
from models.skeleton_model import SkeletonModel
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

wandb.login()

# loading the sweep configuration file
with open(join(dirname(dirname(__file__)), 'skeleton_config.yaml')) as stream:
    config = dict(yaml.safe_load(stream))

train_dataset = Human36M(train=True, heatmap=False)
validation_dataset = Human36M(train=False, heatmap=False)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
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

    hidden_size = wandb.config.hidden_size
    num_layers = wandb.config.num_layers
    dropout = wandb.config.dropout
    lr = wandb.config.lr

    model = SkeletonModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )

    # changing both the encoder and the decoder to 1 fully connected layer
    model.encoder = nn.Linear(34, 128)
    model.rnn = nn.GRU(
        input_size=128,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=False,
        dropout=dropout,
    )
    model.decoder = nn.Linear(hidden_size, 34)

    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, validation_loader)
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', '-c', type=int, default=30)
    parser.add_argument('--epochs', '-e', type=int, default=10)
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

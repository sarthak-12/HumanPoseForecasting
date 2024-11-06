import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from utils.metrics import MPJPE, PDJ

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


class SkeletonModel(pl.LightningModule):
    """
    A skeleton-based pose prediction model with fully connected layers.
    It consists of a 2-layer fully connected encoder, an `nn.GRU` module, and a 2-layer fully connected decoder.

    Args:
        hidden_size (int): The number of features in the hidden state of the `nn.GRU` module. Default: 128

        num_layers (int): Number of layers within the `nn.GRU` module. Default: 1

        dropout: (float): dropout for the `nn.GRU` module. Default: 0.0

        lr (float): learning rate for training. Default: :math:`3e-4`.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 3e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
        )

        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 34),
        )

    # to be used in training without teacher forcing, validation, and evaluation
    def forward(self, sequence: Tensor, samples: int = 10) -> Tensor:
        # forwarding the seed frames through the encoder
        encoded_data = self.encoder(sequence)
        # forwarding the encoded data through the recurrent module and predicts the first frame
        output, hidden_state = self.rnn(encoded_data)
        predictions = []
        predicted_frame = self.decoder(output[-1])
        # residual connection
        predictions.append(predicted_frame + sequence[-1])
        # using the first predicted frame to predict more frames
        for _ in range(samples - 1):
            encoded_data = self.encoder(predictions[-1]).unsqueeze(0)
            output, hidden_state = self.rnn(encoded_data, hidden_state)
            predicted_frame = self.decoder(output[-1])
            predictions.append(predicted_frame + predictions[-1])
        return torch.stack(predictions)

    # to be used in training with teach forcing
    def forward_with_teacher_forcing(self, sequence: Tensor) -> Tensor:
        # forwarding the seed frames and the teacher frames and taking the last 10 outputs as predictions
        encoded_data = self.encoder(sequence)
        output, hidden_state = self.rnn(encoded_data)
        decoded_data = self.decoder(output[-10:])
        predictions = decoded_data + sequence[-10:]
        return predictions

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # data shape is (sequence length, batch size, input size)
        batch = batch.permute(1, 0, 2)
        # seed frames = 10 seed frames + 9 teacher frames
        # last frame can't be used for prediction, but for calculating the loss
        seed_frames, ground_truth = batch[:-1], batch[-10:]
        predictions = self.forward_with_teacher_forcing(seed_frames)
        mse_error = mse_loss(predictions, ground_truth)
        psnr_score = psnr(predictions, ground_truth)
        self.log_dict({
            'MSE training loss': mse_error,
            'PSNR training score': psnr_score,
        })
        # minimze MSE error and maximize PSNR score
        return mse_error - psnr_score

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.permute(1, 0, 2)
        seed_frames, ground_truth = batch[:10], batch[10:]
        predictions = self.forward(seed_frames)
        mse_error = mse_loss(predictions, ground_truth)
        psnr_score = psnr(predictions, ground_truth)
        self.log_dict({
            'MSE validation loss': mse_error,
            'PSNR validation score': psnr_score,
        })
        return mse_error - psnr_score

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.permute(1, 0, 2)
        seed_frames, ground_truth = batch[:10], batch[10:]
        predictions = self.forward(seed_frames)
        mae_error = mae_loss(predictions, ground_truth)
        mse_error = mse_loss(predictions, ground_truth)
        mpjpe_error = MPJPE(predictions, ground_truth)
        pdj_score = PDJ(predictions, ground_truth)
        self.log_dict({
            'MAE evaluation loss': mae_error,
            'MSE evaluation loss': mse_error,
            'MPJPE error': mpjpe_error,
            'PDJ score': pdj_score,
        })
        # minimize all errors and maximize the PDJ
        optimization_metric = mae_error + \
            mse_error + mpjpe_error + (1 - pdj_score)
        self.log('optimization metric', optimization_metric)
        return optimization_metric

    # predicts a sequence of images of length `samples` given a seuqnce of seed images
    def predict(self, seed: Tensor, samples: int) -> Tensor:
        self.eval()
        with torch.no_grad():
            seed = seed.reshape(-1, 1, 34)
            predictions = self.forward(seed, samples)
        return predictions.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

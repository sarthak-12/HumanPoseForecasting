import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils.metrics import MPJPE, PDJ

from .conv_gru import ConvGRU
from .decoder import ResNet18Decoder
from .encoder import ResNet18Encoder

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


class HeatmapModel(pl.LightningModule):
    """
    A heatmap-based pose prediction model with convolutional layers.
    It consists of a `ResNet18`-based encoder, a `ConvGRU` recurrent module, and a `ResNet18`-based decoder.

    Args:
        hidden_channels (int): The number of channels in the hidden state of the `ConvGRU` module. Default: 128

        num_layers (int): Number of layers within the `ConvGRU` module. Default: 1

        dropout: (float): dropout for the `ConvGRU` module. Default: 0.0

        lr (float): learning rate for training. Default: :math:`3e-4`.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 3e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.encoder = ResNet18Encoder()

        self.rnn = ConvGRU(
            input_channels=256,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = ResNet18Decoder(hidden_channels)

    # to be used in training without teacher forcing, validation, and evaluation
    def forward(self, sequence: Tensor, samples: int = 10) -> Tensor:
        l, n, c, h, w = sequence.shape
        # forwarding the seed frames through the encoder
        encoded_data = self.encoder(sequence.reshape(l * n, c, h, w))
        # forwarding the encoded data through the recurrent module and predicts the first frame
        encoded_data = encoded_data.reshape(l, n, *encoded_data.shape[-3:])
        output, hidden_state = self.rnn(encoded_data, device=self.device)
        predictions = []
        predicted_frame = self.decoder(output[-1])
        # residual connection
        predictions.append(predicted_frame + sequence[-1])
        # using the first predicted frame to predict more frames
        for _ in range(samples - 1):
            encoded_data = self.encoder(predictions[-1]).unsqueeze(0)
            output, hidden_state = self.rnn(
                encoded_data, hidden_state, device=self.device)
            predicted_frame = self.decoder(output[-1])
            predictions.append(predicted_frame + predictions[-1])
        return torch.stack(predictions)

    # to be used in training with teach forcing
    def forward_with_teacher_forcing(self, sequence: Tensor) -> Tensor:
        l, n, c, h, w = sequence.shape
        # forwarding the seed frames and the teacher frames and taking the last 10 outputs as predictions
        encoded_data = self.encoder(sequence.reshape(l * n, c, h, w))
        encoded_data = encoded_data.reshape(l, n, *encoded_data.shape[-3:])
        output, hidden_state = self.rnn(encoded_data, device=self.device)
        output = output[-10:].reshape(10 * n, *output.shape[-3:])
        decoded_data = self.decoder(output).reshape(10, n, c, h, w)
        predictions = decoded_data + sequence[-10:]
        return predictions

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # data shape is (sequence length, batch size, channels, height, width)
        batch = batch.permute(1, 0, 2, 3, 4)
        # seed frames = 10 seed frames + 9 teacher frames
        # last frame can't be used for prediction, but for calculating the loss
        seed_frames, ground_truth = batch[:-1], batch[-10:]
        predictions = self.forward_with_teacher_forcing(seed_frames)
        mse_error = mse_loss(predictions, ground_truth)
        psnr_score = psnr(predictions, ground_truth)
        ssim_score = ssim(predictions, ground_truth)
        self.log_dict({
            'MSE training loss': mse_error,
            'PSNR training score': psnr_score,
            'SSIM training score': ssim_score,
        })
        return mse_error + (1 - ssim_score) - psnr_score

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.permute(1, 0, 2, 3, 4)
        seed_frames, ground_truth = batch[:10], batch[10:]
        predictions = self.forward(seed_frames)
        mse_error = mse_loss(predictions, ground_truth)
        psnr_score = psnr(predictions, ground_truth)
        ssim_score = ssim(predictions, ground_truth)
        self.log_dict({
            'MSE validation loss': mse_error,
            'PSNR validation score': psnr_score,
            'SSIM validation score': ssim_score,
        })
        return mse_error + (1 - ssim_score) - psnr_score

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.permute(1, 0, 2, 3, 4)
        seed_frames, ground_truth = batch[:10], batch[10:]
        predictions = self.forward(seed_frames)
        mae_error = mae_loss(predictions, ground_truth)
        mse_error = mse_loss(predictions, ground_truth)
        mpjpe_error = MPJPE(predictions, ground_truth, heatmap=True)
        pdj_score = PDJ(predictions, ground_truth, heatmap=True)
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
            predictions = self.forward(seed.unsqueeze(1), samples)
        return predictions.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

import torch
from torch import Tensor, nn, sigmoid, tanh
from torch.nn.init import calculate_gain, xavier_uniform_


class ConvLSTMCell(nn.Module):
    """
    An implementation of `nn.LSTMCell` with convolutional layers.

    Args:
        input_channels (int): number of channels in the input.
        hidden_channels (int): number of channels in the hidden state.
        dropout (float): probability of applying dropout on the output of the cell. Default: 0.
    """

    def __init__(self, input_channels: int, hidden_channels: int, dropout: float = 0):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # combines all gates into a single layers
        self.layer = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=3,
            padding=1,
        )
        self.dropout = nn.Dropout(dropout)

        xavier_uniform_(self.layer.weight, gain=calculate_gain('sigmoid'))

    def forward(self, sequence: Tensor, states: 'tuple[Tensor, Tensor]') -> 'tuple[Tensor, tuple[Tensor, Tensor]]':
        hidden_state, cell_state = states
        output = []
        # process each frame sequentially
        for frame in sequence:
            input = torch.cat((frame, hidden_state), dim=1)
            # splits into forget, input, cell, and output gates
            gates = self.layer(input).chunk(4, dim=1)
            forget_gate = sigmoid(gates[0])
            input_gate = sigmoid(gates[1])
            cell_gate = tanh(gates[2])
            output_gate = sigmoid(gates[3])
            cell_state = forget_gate * cell_state + input_gate * cell_gate
            hidden_state = output_gate * tanh(cell_state)
            output.append(hidden_state)
        output = self.dropout(torch.stack(output))
        return output, (hidden_state, cell_state)


class ConvLSTM(nn.Module):
    """
    An implementation of `nn.LSTM` with convolutional layers.

    Args:
        input_channels (int): number of channels in the input.

        input_size (int): size of the image/frame in the input.

        hidden_channels (int): number of channels in the hidden state.

        num_layers (int): number of layers/cells. Default: 1

        dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        dropout = 0 if num_layers == 1 else dropout
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.layers = [ConvLSTMCell(input_channels, hidden_channels, dropout)]
        for _ in range(num_layers - 1):
            self.layers.append(ConvLSTMCell(
                hidden_channels, hidden_channels, dropout))
        # disable the dropout in the last layer
        self.layers[-1].dropout = nn.Identity()
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: Tensor, states: 'tuple[Tensor, Tensor]' = None) -> 'tuple[Tensor, tuple[Tensor, Tensor]]':
        # initializes the hidden state and the cell if not provided
        n = x.shape[1]
        h = x.shape[-2]
        w = x.shape[-1]
        if states is None:
            states = (
                torch.zeros(
                    (self.num_layers, n, self.hidden_channels, h, w,),
                    device=self.device,
                ),
                torch.zeros(
                    (self.num_layers, n, self.hidden_channels, h, w,),
                    device=self.device,
                ),
            )
        hidden_state, cell_state = states
        # forwarding input through the layers and updating the hidden state
        output = x
        new_hidden_state = hidden_state.clone()
        new_cell_state = cell_state.clone()
        for i, layer in enumerate(self.layers):
            h = hidden_state[i]
            c = cell_state[i]
            output, (h, c) = layer(output, (h, c))
            new_hidden_state[i] = h
            new_cell_state[i] = c
        return output, (new_hidden_state, new_cell_state)

import torch
from torch import Tensor, nn, sigmoid, tanh
from torch.nn.init import calculate_gain, xavier_uniform_


class ConvGRUCell(nn.Module):
    """
    An implementation of `nn.GRUCell` with convolutional layers.

    Args:
        input_channels (int): number of channels in the input.
        hidden_channels (int): number of channels in the hidden state.
        dropout (float): probability of applying dropout on the output of the cell. Default: 0.
    """

    def __init__(self, input_channels: int, hidden_channels: int, dropout: float = 0):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # combines the reset and update gates.
        self.layer1 = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        # the new gate
        self.layer2 = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        xavier_uniform_(self.layer1.weight, gain=calculate_gain('sigmoid'))
        xavier_uniform_(self.layer2.weight, gain=calculate_gain('tanh'))

    def forward(self, sequence: Tensor, hidden_state: Tensor) -> 'tuple[Tensor, Tensor]':
        output = []
        # process each frame sequentially
        for frame in sequence:
            input = torch.cat((frame, hidden_state), dim=1)
            # splits into reset and update gates
            reset_gate, update_gate = sigmoid(
                self.layer1(input)).chunk(2, dim=1)
            new_gate_input = torch.cat(
                (frame, reset_gate * hidden_state), dim=1)
            new_gate = tanh(self.layer2(new_gate_input))
            hidden_state = (1 - update_gate) * \
                new_gate + update_gate * hidden_state
            output.append(hidden_state)
        output = self.dropout(torch.stack(output))
        return output, hidden_state


class ConvGRU(nn.Module):
    """
    An implementation of `nn.GRU` with convolutional layers.

    Args:
        input_channels (int): number of channels in the input.

        input_size (int): size of the image/frame in the input.

        hidden_channels (int): number of channels in the hidden state.

        num_layers (int): number of layers/cells. Default: 1

        dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
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
        self.layers = [ConvGRUCell(input_channels, hidden_channels, dropout)]
        for _ in range(num_layers - 1):
            self.layers.append(ConvGRUCell(
                hidden_channels, hidden_channels, dropout))
        # disable the dropout in the last layer
        self.layers[-1].dropout = nn.Identity()
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: Tensor, hidden_state: Tensor = None, device: str = 'cpu') -> 'tuple[Tensor, Tensor]':
        # initializes the hidden state if not provided
        n = x.shape[1]
        h = x.shape[-2]
        w = x.shape[-1]
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_layers, n, self.hidden_channels, h, w,),
                device=device,
            )
        # forwarding input through the layers and updating the hidden state
        output = x
        new_hidden_state = hidden_state.clone()
        for i, layer in enumerate(self.layers):
            h = hidden_state[i]
            output, h = layer(output, h)
            new_hidden_state[i] = h
        return output, new_hidden_state

from torch import nn, Tensor


class BasicBlock(nn.Module):
    """
    A simplified and reveresed version of resnet's `BasicBlock`, contains only the components used in ResNet18.

    Args:
        in_channels (int): Number of channels in the input image.

        out_channels (int): Number of channels produced by the convolution.

        stride (int): Stride of the first convolutional layer. Default: 1.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(
            out_planes,
            out_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                output_padding=stride // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride > 1 or self.in_planes != self.out_planes:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Decoder(nn.Module):
    """
    A mirrored version of the modified ResNet18-based encoder.

    Args:
        hidden_channels (int): Number of channels in the input.
    """

    def __init__(self, hidden_channels: int = 128):
        super().__init__()
        self.layer1 = self._make_layer(hidden_channels, 128, stride=2)
        self.layer2 = self._make_layer(128, 64, stride=2)
        self.layer3 = self._make_layer(64, 64)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=17,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
        layers = nn.Sequential(
            BasicBlock(in_planes, out_planes, stride),
            BasicBlock(out_planes, out_planes),
        )
        return layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv1(x)
        return x

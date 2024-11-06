from torch import Tensor, nn
from torchvision import models


class ResNet18Encoder(nn.Module):
    """
    A modified version of the ResNet18 network as an encoder. This encoder skips `layer4`, `maxpool`, `avgpool`, and `fc` layers, and changes the `in_channels` and `stride` parameters of the `conv1` layer to more suitable values.
    """

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(
            in_channels=17,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        # skipping layer4 shrinks down the model size by almost 75%
        self.model.layer4 = nn.Identity()
        # skipping the pooling layers and the fully connected layer to maintain a desired shape for the output
        self.model.maxpool = nn.Identity()
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # stride = 1 if the decoder is not used
        scale_factor = self.model.layer2[0].stride * \
            self.model.layer3[0].stride
        n = x.shape[0]
        c = self.model.layer3[0].conv1.out_channels
        h = x.shape[-2] // scale_factor
        w = x.shape[-1] // scale_factor
        # undoing the torch.flatten() operation at the end of the original `forward` definition
        out = self.model(x).reshape((n, c, h, w))
        return out

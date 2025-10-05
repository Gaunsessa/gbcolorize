import torch
import torch.nn as nn
import torchvision.models as models

from torchtyping import TensorType


class RespModel(nn.Module):
    def __init__(self):
        super(RespModel, self).__init__()

        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # self.frozen = False

        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    resnet34.conv1,
                    resnet34.bn1,
                    resnet34.relu,
                ),
                nn.Sequential(
                    resnet34.maxpool,
                    resnet34.layer1,
                ),
                resnet34.layer2,
            ]
        )

        self.bottleneck = resnet34.layer3

        self.encoder_params = [
            p
            for p in [*self.encoder.parameters(), *self.bottleneck.parameters()]
            if p.requires_grad
        ]

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            ]
        )

    def forward(
        self, x: TensorType["batch", 1, 112, 128]
    ) -> TensorType["batch", 2, 112, 128]:
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std

        encodes = []
        for encoder in self.encoder:
            x = encoder(x)
            encodes.append(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
            x = decoder(x)

            if i != len(self.decoder) - 1:
                # x = torch.cat([x, encodes[-i - 1]], dim=1)
                x = x + encodes[-i - 1]

        return x

    # def train(self, mode: bool = True):
    #     self.training = mode

    #     self.encoder.train(mode)

    #     self.decoder.train(mode and not self.frozen)
    #     self.bottleneck.train(mode and not self.frozen)

    # def eval(self):
    #     self.training = False

    #     self.encoder.eval()
    #     self.bottleneck.eval()
    #     self.decoder.eval()

    def freeze_encoder(self, freeze: bool = True):
        # self.frozen = freeze

        # self.encoder.train(not freeze)
        # self.bottleneck.train(not freeze)

        for param in self.encoder_params:
            param.requires_grad = not freeze

    def init_weights(self):
        for layer in self.decoder.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                # nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

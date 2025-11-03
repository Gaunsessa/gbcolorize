import torch
import torch.nn as nn
import torchvision.models as models


class RespModel(nn.Module):
    def __init__(self):
        super(RespModel, self).__init__()

        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # self.frozen = False

        self.mean = (0.229 + 0.224 + 0.225) / 3
        self.std = (0.229 + 0.224 + 0.225) / 3

        self.expand = nn.Conv2d(1, 3, 5, stride=1, padding=2)

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
                    nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand((x - self.mean) / self.std)

        encodes = []
        for encoder in self.encoder:
            x = encoder(x)
            encodes.append(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
            x = decoder(x)

            if i != len(self.decoder) - 1:
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

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

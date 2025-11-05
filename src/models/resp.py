import torch
import torch.nn as nn
import torch.nn.functional as tf
import torchvision.models as models


class RespModel(nn.Module):
    def __init__(self):
        super(RespModel, self).__init__()

        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.expand = nn.Conv2d(1, 3, 1, stride=1)

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
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(64, 256, 4, stride=2, padding=1),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        encodes = []
        for encoder in self.encoder:
            x = encoder(x)
            encodes.append(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
            print(i)
            x = decoder(x)

            if i != len(self.decoder) - 1:
                x = x + encodes[-i - 1]
                x = tf.leaky_relu(x)

        return x

    def freeze_encoder(self, freeze: bool = True):
        for param in self.encoder_params:
            param.requires_grad = not freeze

    def init_weights(self):
        self.expand.weight.data.fill_(1)
        if self.expand.bias is not None:
            self.expand.bias.data.fill_(0)
        
        for layer in self.decoder.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

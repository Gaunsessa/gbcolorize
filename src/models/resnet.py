import torch
import torch.nn as nn
import torch.nn.functional as tf

from torchvision.models import resnet34, ResNet34_Weights

from lightning import LightningModule


class ResnetModel(LightningModule):
    def __init__(self, output_features: int, loss_fn: torch.nn.Module, lr: float):
        super().__init__()

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.expand = nn.Conv2d(1, 3, 1, stride=1)

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                ),
                nn.Sequential(
                    resnet.maxpool,
                    resnet.layer1,
                ),
                resnet.layer2,
            ]
        )

        self.bottleneck = resnet.layer3

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
                    nn.ConvTranspose2d(64, output_features, 4, stride=2, padding=1),
                ),
            ]
        )

        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        encodes = []
        for encoder in self.encoder:
            x = encoder(x)
            encodes.append(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
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

    # Training
    def training_step(self, batch, batch_idx):
        input, target = batch

        pred = self.forward(input / 3.0)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

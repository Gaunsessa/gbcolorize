import torch
import torch.nn as nn
import torch.nn.functional as tf

from lightning import LightningModule

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
)

from .base import BaseModel


class EfficientNetEncoder(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        sizes = [
            (efficientnet_b0, EfficientNet_B0_Weights),
            (efficientnet_b1, EfficientNet_B1_Weights),
            (efficientnet_b2, EfficientNet_B2_Weights),
            (efficientnet_b3, EfficientNet_B3_Weights),
        ]

        model = sizes[size][0](weights=sizes[size][1].DEFAULT)

        self.stem = model.features[0]
        self.stage1 = model.features[1]
        self.stage2 = model.features[2]
        self.stage3 = model.features[3]
        self.stage4 = model.features[4]
        self.stage5 = model.features[5]
        self.stage6 = model.features[6]
        self.stage7 = model.features[7]
        # self.head   = model.features[8]

        self.output_features = [
            self.stage1[-1].out_channels,
            self.stage2[-1].out_channels,
            self.stage3[-1].out_channels,
            self.stage5[-1].out_channels,
            self.stage7[-1].out_channels,
        ]

    def forward(self, x):
        skips = []

        x = self.stem(x)
        x = self.stage1(x)
        skips.append(x)

        x = self.stage2(x)
        skips.append(x)

        x = self.stage3(x)
        skips.append(x)

        x = self.stage4(x)

        x = self.stage5(x)
        skips.append(x)

        x = self.stage6(x)

        x = self.stage7(x)
        skips.append(x)

        return x, skips

    def freeze(self, freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = not freeze


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        output_activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            output_activation,
        )

    def forward(self, x, skip):
        x = tf.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)

        x = self.block1(x)
        x = self.block2(x)

        return x

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


class EfficientModel(LightningModule):
    def __init__(self, size: int, output_features: int, loss_fn: torch.nn.Module):
        super().__init__()

        self.expand = nn.Conv2d(1, 3, 1, stride=1, padding=0)

        self.encoder = EfficientNetEncoder(size)

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(self.encoder.output_features[4], 256, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.decode1 = DecoderBlock(256, self.encoder.output_features[3], 128)
        self.decode2 = DecoderBlock(128, self.encoder.output_features[2], 64)
        self.decode3 = DecoderBlock(64, self.encoder.output_features[1], 32)
        self.decode4 = DecoderBlock(32, self.encoder.output_features[0], 16)

        self.output = DecoderBlock(
            16, 1, output_features, output_activation=nn.Identity()
        )

        self.loss_fn = loss_fn

    def forward(self, input):
        with torch.no_grad():
            x = input.repeat(1, 3, 1, 1)
            x, skips = self.encoder(x)

        x = self.bottle_neck(x)

        x = self.decode1(x, skips[3])
        x = self.decode2(x, skips[2])
        x = self.decode3(x, skips[1])
        x = self.decode4(x, skips[0])

        x = self.output(x, input)

        return x

    def training_step(self, batch, batch_idx):
        input, target = batch

        pred = self.forward(input / 3.0)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)

        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def freeze_encoder(self, freeze: bool = True):
        self.encoder.freeze(freeze)

    def init_weights(self):
        self.expand.weight.data.fill_(1)
        if self.expand.bias is not None:
            self.expand.bias.data.fill_(0)

        self.decode1.init_weights()
        self.decode2.init_weights()
        self.decode3.init_weights()
        self.decode4.init_weights()
        self.output.init_weights()

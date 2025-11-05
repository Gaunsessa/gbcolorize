import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as tf


class EfficientNetB3Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        model = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        self.stem = model.features[0]
        self.stage1 = model.features[1:3]
        self.stage2 = model.features[3:5]
        self.stage3 = model.features[5:7]
        self.stage4 = model.features[7:10]
        self.stage5 = model.features[10:13]
        self.stage6 = model.features[13:17]
        self.stage7 = model.features[17:]

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
        skips.append(x)

        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)

        return x, skips


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        output_activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            output_activation,
        )

    def forward(self, x, skip):
        x = tf.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)

        x = self.block1(x)
        x = self.block2(x)

        return x


class EfficientModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand = nn.Conv2d(1, 3, 1, stride=1, padding=0)

        self.encoder = EfficientNetB3Encoder()

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(1536, 512, 4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.decode1 = DecoderBlock(512, 1536, 512)
        self.decode2 = DecoderBlock(512, 232, 512)
        self.decode3 = DecoderBlock(512, 96, 256)
        self.decode4 = DecoderBlock(256, 32, 256)

        self.output = DecoderBlock(256, 1, 256, output_activation=nn.Identity())

    def forward(self, input):
        x = self.expand(input)

        x, skips = self.encoder(x)

        x = self.bottle_neck(x)

        x = self.decode1(x, skips[3])
        x = self.decode2(x, skips[2])
        x = self.decode3(x, skips[1])
        x = self.decode4(x, skips[0])

        x = self.output(x, input)

        return x

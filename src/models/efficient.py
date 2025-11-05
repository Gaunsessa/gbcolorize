import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as tf


class EfficientNetB3Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        model = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT
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

    def freeze(self, freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = not freeze


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        output_activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.skip_reduce = nn.Conv2d(skip_channels, min(in_channels, skip_channels), kernel_size=1, bias=False)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels + min(in_channels, skip_channels),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            output_activation,
        )

    def forward(self, x, skip):
        x = tf.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        
        skip = self.skip_reduce(skip)

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


class EfficientModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand = nn.Conv2d(1, 3, 1, stride=1, padding=0)

        self.encoder = EfficientNetB3Encoder()

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(1536, 512, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 256, 4, stride=1, padding=0),
            nn.ReLU(),
        )

        self.decode1 = DecoderBlock(256, 1536, 256)
        self.decode2 = DecoderBlock(256, 232, 128)
        self.decode3 = DecoderBlock(128, 96, 128)
        self.decode4 = DecoderBlock(128, 32, 256)

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

import torch
import torch.nn as nn
import torch.nn.functional as tf

from torchtyping import TensorType

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def init_weights(self):
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoders = nn.ModuleList([
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        ])

        self.bottleneck = ConvBlock(128, 128)

        self.decoders = nn.ModuleList([
            ConvBlock(256, 64),
            ConvBlock(128, 32),
            ConvBlock(64, 16),
            ConvBlock(32, 2),
        ])

        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ConvTranspose2d(16, 16, 2, stride=2),
        ])

    def forward(self, x: TensorType["batch", 1, 112, 128]) -> TensorType["batch", 2, 112, 128]:
        encodes = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encodes.append(x)
            x = tf.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = self.up_samples[i](x)
            x = torch.cat([x, encodes[-i - 1]], dim=1)
            x = decoder(x)
        
        return x
    
    def init_weights(self):
        for encoder in self.encoders:
            encoder.init_weights()

        self.bottleneck.init_weights()

        for decoder in self.decoders:
            decoder.init_weights()

        for up_sample in self.up_samples:
            nn.init.kaiming_normal_(up_sample.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(up_sample.bias, 0)
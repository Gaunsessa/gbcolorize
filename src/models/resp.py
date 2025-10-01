import torch
import torch.nn as nn
import torchvision.models as models

from torchtyping import TensorType

class RespModel(nn.Module):
    def __init__(self):
        super(RespModel, self).__init__()

        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3,
            # resnet34.layer4,
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
        )

    def forward(self, x: TensorType["batch", 1, 112, 128]) -> TensorType["batch", 2, 112, 128]:
        x = self.encoder(x.expand(-1, 3, -1, -1))
        x = self.decoder(x)

        return x

    def init_weights(self):
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
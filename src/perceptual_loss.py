import torch
import torch.nn as nn

from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg16.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return torch.nn.functional.mse_loss(self.vgg16(x), self.vgg16(y))

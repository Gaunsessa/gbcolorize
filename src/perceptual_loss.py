import torch
import torch.nn as nn

from torchvision import models

from utils.color import lab_to_rgb


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg16.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x_input, x_pred, y_input, y_pred):
        x = torch.cat([x_input, x_pred], dim=1)
        y = torch.cat([y_input, y_pred], dim=1)

        x = torch.vmap(lab_to_rgb)(x.view(x.shape[0], 3, -1)).view(x.shape[0], 3, 112, 128)
        y = torch.vmap(lab_to_rgb)(y.view(y.shape[0], 3, -1)).view(y.shape[0], 3, 112, 128)

        return torch.nn.functional.mse_loss(self.vgg16(x), self.vgg16(y))

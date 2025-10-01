import torch
import torch.nn as nn

from torchvision import models

from utils.color import lab_to_rgb


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg16.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, grey, x_ab, y_ab):
        grey[grey == 0] = 0.60
        grey[grey == 1] = 0.83
        grey[grey == 2] = 0.91
        grey[grey == 3] = 0.97

        x = torch.cat([grey, x_ab], dim=1)
        y = torch.cat([grey, y_ab], dim=1)

        x = torch.vmap(lab_to_rgb)(x.view(x.shape[0], 3, -1)).view(x.shape[0], 3, 112, 128)
        y = torch.vmap(lab_to_rgb)(y.view(y.shape[0], 3, -1)).view(y.shape[0], 3, 112, 128)

        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x = (x - mean) / std
        y = (y - mean) / std

        return torch.nn.functional.mse_loss(self.vgg16(x), self.vgg16(y))

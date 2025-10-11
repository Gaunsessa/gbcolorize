import torch
import torch.nn as nn

from torchvision import models

from utils.color import lab_to_rgb


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        self.selected_layers = [3, 8, 15]
        self.vgg16 = nn.Sequential(*list(vgg16)[:16])
        self.vgg16.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, grey, x_ab, y_ab):
        grey = grey.clone()
        
        grey[grey == 0] = 0.60
        grey[grey == 1] = 0.83
        grey[grey == 2] = 0.91
        grey[grey == 3] = 0.97

        x = torch.cat([grey, x_ab], dim=1)
        y = torch.cat([grey, y_ab], dim=1)

        x = torch.vmap(lab_to_rgb)(x.view(x.shape[0], 3, -1)).view(x.shape[0], 3, 112, 128)
        y = torch.vmap(lab_to_rgb)(y.view(y.shape[0], 3, -1)).view(y.shape[0], 3, 112, 128)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0

        for i, layer in enumerate(self.vgg16):
            x = layer(x)
            y = layer(y)
            
            if i in self.selected_layers:
                loss += nn.functional.mse_loss(x, y)

        return loss

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

    def forward(self, grey, pred, target):
        grey = grey.clone()
        
        grey[grey == 0] = 0.60
        grey[grey == 1] = 0.83
        grey[grey == 2] = 0.91
        grey[grey == 3] = 0.97

        pred = torch.cat([grey, pred], dim=1)
        target = torch.cat([grey, target], dim=1)

        pred = torch.vmap(lab_to_rgb)(pred.view(pred.shape[0], 3, -1)).view(pred.shape[0], 3, 112, 128)
        target = torch.vmap(lab_to_rgb)(target.view(target.shape[0], 3, -1)).view(target.shape[0], 3, 112, 128)

        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0

        for i, layer in enumerate(self.vgg16):
            pred = layer(pred)

            with torch.no_grad():
                target = layer(target)
            
            if i in self.selected_layers:
                loss += nn.functional.mse_loss(pred, target)

        return loss

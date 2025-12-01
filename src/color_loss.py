import torch
import torch.nn as nn
import torch.nn.functional as tf


class BinnedColorLoss(nn.Module):
    knn_idx: torch.Tensor
    knn_weights: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        knn_idx: torch.Tensor,
        knn_weights: torch.Tensor,
        weights: torch.Tensor,
    ):
        super().__init__()

        self.register_buffer("knn_idx", knn_idx)
        self.register_buffer("knn_weights", knn_weights)
        self.register_buffer("weights", weights)

    def forward(self, pred, _color, binned_color):
        prob = tf.log_softmax(pred, dim=1)
        
        target_idx = binned_color.squeeze(1)

        idx = self.knn_idx[target_idx]
        wts = self.knn_weights[target_idx]
        weights = self.weights[target_idx]

        idx = idx.permute(0, 3, 1, 2)
        wts = wts.permute(0, 3, 1, 2)

        p = torch.gather(prob, 1, idx)

        loss = -((p * wts).sum(dim=1) * weights).mean()

        return loss

class ColorLoss(nn.Module):
    weights: torch.Tensor

    def __init__(self, weights: torch.Tensor):
        super().__init__()

        self.register_buffer("weights", weights)

    def forward(self, pred, color, binned_color):
        weights = self.weights[binned_color]

        loss = tf.mse_loss(pred, color, reduction="none")
        loss = (loss * weights).mean()

        return loss
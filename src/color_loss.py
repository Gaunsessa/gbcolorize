import torch
import torch.nn as nn
import torch.nn.functional as tf


class ColorLoss(nn.Module):
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

    def forward(self, pred, target):
        prob = tf.log_softmax(pred, dim=1)
        
        target_idx = target.squeeze(1)

        idx = self.knn_idx[target_idx]
        wts = self.knn_weights[target_idx]
        weights = self.weights[target_idx]

        idx = idx.permute(0, 3, 1, 2)
        wts = wts.permute(0, 3, 1, 2)

        p = torch.gather(prob, 1, idx)

        loss = -((p * wts).sum(dim=1) * weights).mean()

        return loss

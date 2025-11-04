import torch
import torch.nn.functional as tf


def cross_entropy_color_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_weights: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    prob = tf.log_softmax(pred, dim=1)

    target_idx = target.squeeze(1)

    idx = knn_idx[target_idx]
    wts = knn_weights[target_idx]

    weights = weights[target_idx]

    idx = idx.permute(0, 3, 1, 2)
    wts = wts.permute(0, 3, 1, 2)

    p = torch.gather(prob, 1, idx)

    loss = -((p * wts).sum(dim=1) * weights).mean()

    return loss

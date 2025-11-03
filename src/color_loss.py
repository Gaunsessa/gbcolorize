import torch
import torch.nn.functional as tf


def cross_entropy_color_loss(pred, target, knn_idx, knn_weights):
    prob = tf.log_softmax(pred, dim=1)

    target_idx = target.squeeze(1)
    idx = knn_idx[target_idx]
    wts = knn_weights[target_idx]

    idx = idx.permute(0, 3, 1, 2)
    wts = wts.permute(0, 3, 1, 2)

    p = torch.gather(prob, 1, idx)

    loss = -(p * wts).sum(dim=1).mean()
    return loss

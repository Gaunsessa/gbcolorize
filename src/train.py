import os
import argparse

import torch

import numpy as np

from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from dataloader import GBColorizeDataset
from color_loss import ColorLoss

from utils.color import get_color_bins, precompute_color_bins_weights

from models.efficient2 import EfficientModel


def load_dataset(path: str):
    luma = torch.empty(0, 1, 112, 128, dtype=torch.float16)
    color = torch.empty(0, 1, 112, 128, dtype=torch.long)

    paths = sorted(
        [
            os.path.join(path, chunk)
            for chunk in os.listdir(path)
            if chunk.endswith(".npz")
        ]
    )

    for path in paths:
        data = np.load(path)
        luma = torch.cat((luma, torch.tensor(data["luma"])), dim=0)
        color = torch.cat((color, torch.tensor(data["color"], dtype=torch.long)), dim=0)

    return luma, color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    bins = get_color_bins()
    bin_knn_idx, bin_knn_weights = precompute_color_bins_weights(bins)
    bin_weights = torch.load(os.path.join(args.dataset, "bin_weights.pt"))

    loss_fn = ColorLoss(bin_knn_idx, bin_knn_weights, bin_weights)

    model = EfficientModel(0, len(bins), loss_fn)
    model.init_weights()
    model.freeze_encoder()
    
    luma, color = load_dataset(args.dataset)
    luma.share_memory_()
    color.share_memory_()
    
    dataset = GBColorizeDataset(luma, color)
    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4, persistent_workers=True
    )

    logger = TensorBoardLogger(save_dir="runs")
    trainer = Trainer(logger=logger)
    
    trainer.fit(model=model, train_dataloaders=dataloader)

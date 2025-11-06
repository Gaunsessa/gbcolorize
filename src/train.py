import os
import argparse

import torch

import numpy as np

from torch.utils.data import DataLoader

from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from tqdm import tqdm

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

    for path in tqdm(paths, desc="Loading dataset"):
        data = np.load(path)
        luma = torch.cat((luma, torch.tensor(data["luma"])), dim=0)
        color = torch.cat((color, torch.tensor(data["color"], dtype=torch.long)), dim=0)

    luma.share_memory_()
    color.share_memory_()

    return luma, color


class GBColorizeDataModule(LightningDataModule):
    def __init__(self, dataset_path: str, batch_size=64, num_workers=4):
        super().__init__()
        self.luma, self.color = load_dataset(dataset_path)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Wrap shared memory tensors into your Dataset
        self.dataset = GBColorizeDataset(self.luma, self.color)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )


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

    datamodule = GBColorizeDataModule(args.dataset)

    logger = TensorBoardLogger(save_dir="runs")
    trainer = Trainer(logger=logger, strategy=DDPStrategy(find_unused_parameters=True))

    trainer.fit(model=model, datamodule=datamodule)

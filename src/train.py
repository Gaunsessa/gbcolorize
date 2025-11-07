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

from models.resnet import ResnetModel
from models.efficient import EfficientModel


MODELS = {
    "resnet": ResnetModel,
    "efficient": EfficientModel,
}


class GBColorizeDataModule(LightningDataModule):
    def __init__(self, dataset_path: str, batch_size=64, num_workers=4):
        super().__init__()

        self.load_dataset(dataset_path)
        self.dataset = GBColorizeDataset(self.luma, self.color)

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.dataset, [0.95, 0.05]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_dataset(self, path: str):
        self.luma = torch.empty(0, 1, 112, 128, dtype=torch.float16)
        self.color = torch.empty(0, 1, 112, 128, dtype=torch.long)

        paths = [
            os.path.join(path, chunk)
            for chunk in os.listdir(path)
            if chunk.endswith(".npz")
        ]

        for path in tqdm(paths, desc="Loading dataset"):
            data = np.load(path)
            self.luma = torch.cat((self.luma, torch.tensor(data["luma"])), dim=0)
            self.color = torch.cat(
                (self.color, torch.tensor(data["color"], dtype=torch.long)), dim=0
            )

        self.luma.share_memory_()
        self.color.share_memory_()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys())
    parser.add_argument("--size", type=int, required=False, choices=[0, 1, 2, 3])
    parser.add_argument("--binned", type=bool, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--weight_alpha", type=float, required=False, default=1.0)

    args = parser.parse_args()

    bins = get_color_bins()
    bin_knn_idx, bin_knn_weights = precompute_color_bins_weights(bins)
    bin_weights = torch.load(os.path.join(args.dataset, "bin_weights.pt"))

    loss_fn = ColorLoss(bin_knn_idx, bin_knn_weights, bin_weights**args.weight_alpha)

    model = EfficientModel(args.size, len(bins), loss_fn, lr=args.lr)
    model.init_weights()
    model.freeze_encoder()

    datamodule = GBColorizeDataModule(args.dataset, batch_size=args.batch)

    logger = TensorBoardLogger(save_dir="runs")
    trainer = Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True, start_method="spawn"),
        precision="16-mixed",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

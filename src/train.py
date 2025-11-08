import os
import argparse
from typing import MutableMapping

from lightning.fabric.utilities.rank_zero import rank_zero_only
import torch

from torch.utils.data import DataLoader

from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

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
    def __init__(
        self,
        dataset_path: str,
        bins: torch.Tensor,
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()

        self.dataset = GBColorizeDataset(os.path.join(dataset_path, "imgs/"), bins)

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.dataset, [0.95, 0.05]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

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
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


# https://stackoverflow.com/a/70704227
class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        assert isinstance(metrics, MutableMapping)

        metrics.pop("epoch", None)

        return super().log_metrics(metrics, step)


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

    datamodule = GBColorizeDataModule(args.dataset, bins, batch_size=args.batch)

    logger = TBLogger(name=None, save_dir="runs", default_hp_metric=False)
    trainer = Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

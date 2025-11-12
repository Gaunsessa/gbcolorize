import os
import torch
import argparse

from typing import MutableMapping

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy

from dataloader import GBColorizeDataModule
from color_loss import ColorLoss

from utils.color import get_color_bins, precompute_color_bins_weights

from models.resnet import ResnetModel
from models.efficient import EfficientModel


MODELS = {
    "resnet": ResnetModel,
    "efficient": EfficientModel,
}


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
    parser.add_argument("--workers", type=int, required=False, default=1)

    args = parser.parse_args()

    bins = get_color_bins()
    bin_knn_idx, bin_knn_weights = precompute_color_bins_weights(bins)
    bin_weights = torch.load(os.path.join(args.dataset, "bin_weights.pt"))

    loss_fn = ColorLoss(bin_knn_idx, bin_knn_weights, bin_weights**args.weight_alpha)

    model = EfficientModel(args.size, len(bins), loss_fn, lr=args.lr)
    model.init_weights()
    model.freeze_encoder()

    datamodule = GBColorizeDataModule(os.path.join(args.dataset, "data"), args.batch, args.workers)

    logger = TBLogger(name=None, save_dir="runs", default_hp_metric=False)
    trainer = Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

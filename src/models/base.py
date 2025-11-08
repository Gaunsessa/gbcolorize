import torch
import torch.nn as nn

from abc import ABC

from lightning import LightningModule

from utils.color import dequantize_colors, get_color_bins, vlab_to_rgb


class BaseModel(LightningModule, ABC):
    output_features: int
    loss_fn: nn.Module
    lr: float
    
    def __init__(self, output_features: int, loss_fn: nn.Module, lr: float):
        super().__init__()
        self.save_hyperparameters()
        
        self.output_features = output_features
        self.loss_fn = loss_fn
        self.lr = lr

    def training_step(self, batch, batch_idx):
        input, target = batch

        pred = self.forward(input / 3.0)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch

        pred = self.forward(input / 3.0)

        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss)

        if batch_idx == 0:
            pred = pred[:100].cpu()
            input = input[:100].cpu()

            luma_mapping = torch.tensor([0.60, 0.83, 0.91, 0.97], device=input.device)
            luma = luma_mapping[input]

            color = pred.argmax(dim=1, keepdim=True)

            imgs = torch.cat([luma, color], dim=1)
            imgs = dequantize_colors(imgs, get_color_bins())
            imgs = vlab_to_rgb(imgs)

            # I do it like this to make pyright happy ;-;
            getattr(self.logger, "experiment").add_images(
                "val_images", imgs, self.current_epoch
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

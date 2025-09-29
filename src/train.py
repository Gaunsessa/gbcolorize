import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as tf

import numpy as np

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataloader import GBColorizeDataset
from models.conv import GBConvModel
from models.unet import UNet

from utils.color import vlab_to_rgb, vrgb_to_lab


MODELS = {
    "unet": UNet,
    "conv": GBConvModel,
}


class Trainer:
    model: nn.Module

    def __init__(self, model, optim, name, device):
        self.model = model
        self.optim = optim
        self.device = device
        self.writer = SummaryWriter()
        self.name = name

        self.epoch = 0
        self.steps = 0

    def forward_epoch(self, dl):
        self.model.train()

        for input, target in tqdm(dl, desc=f"Epoch {self.epoch}", total=len(dl)):
            input = input.to(self.device)
            target = target.to(self.device)

            pred = self.model.forward(input)

            loss = tf.l1_loss(pred, target)
            self.writer.add_scalar("Loss/train", loss.item(), self.steps)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.steps += 1

    def forward_validate(self, dl):
        self.model.eval()

        input, target = next(iter(dl))
        input = input.to(device)
        target = target.to(device)

        pred = model.forward(input)
        loss = tf.l1_loss(pred, target)

        input[:, 0][input[:, 0] == 0] = 0.60
        input[:, 0][input[:, 0] == 1] = 0.83
        input[:, 0][input[:, 0] == 2] = 0.91
        input[:, 0][input[:, 0] == 3] = 0.97

        imgs = torch.cat([input, pred], dim=1)[:100]
        imgs = vlab_to_rgb(imgs)

        self.writer.add_scalar("Loss/val", loss.item(), self.epoch)
        self.writer.add_images(f"Pred Images", imgs, self.epoch)

    def checkpoint(self):
        os.makedirs(f"ckpts/{self.name}", exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "epoch": self.epoch,
            },
            f"ckpts/{self.name}/epoch_{self.epoch}.pth",
        )

    def train(self, epochs, train_dl, val_dl):
        for self.epoch in range(epochs):
            self.forward_epoch(train_dl)
            self.forward_validate(val_dl)

            if self.epoch % 10 == 0 or self.epoch == epochs - 1:
                self.checkpoint()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python train.py <model> <dataset> <epochs> <batch_size> <lr>")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset = sys.argv[2]
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    lr = float(sys.argv[5])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = torch.tensor(
        np.concatenate(
            [
                np.load(os.path.join(dataset, path))["imgs"]
                for path in tqdm(os.listdir(dataset), desc="Loading dataset")
                if path.endswith(".npz")
            ]
        ),
        dtype=torch.uint8,
    )

    dataset.share_memory_()

    ds = GBColorizeDataset(dataset)

    train_ds, val_ds = random_split(ds, [0.9, 0.1])

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=8, persistent_workers=True, pin_memory=True
    )

    val_dl = DataLoader(val_ds, batch_size=len(val_ds))

    model = MODELS[model_name]()
    model.init_weights()
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    run_name = f"{model_name}_{dataset}_{batch_size}_{lr}_{epochs}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)

    trainer = Trainer(model, optim, run_name, device)
    trainer.train(epochs, train_dl, val_dl)

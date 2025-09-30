import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataloader import GBColorizeDataset
from models.conv import GBConvModel
from models.unet import UNet

from utils.color import rgb_to_lab, vlab_to_rgb


MODELS = {
    "unet": UNet,
    "conv": GBConvModel,
}


class Trainer:
    model: nn.Module

    def __init__(self, model, optim, name, device, rank):
        self.model = model
        self.optim = optim
        self.device = device
        self.writer = SummaryWriter() if rank == 0 else None
        self.name = name
        self.rank = rank

        self.epoch = 0
        self.steps = 0

    def forward_epoch(self, dl):
        self.model.train()

        for input, target in tqdm(dl, desc=f"Epoch {self.epoch}", total=len(dl), disable=self.rank != 0):
            pred = self.model.forward(input)

            loss = tf.l1_loss(pred, target)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", loss.item(), self.steps)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.steps += 1

    def forward_validate(self, dl):
        self.model.eval()

        loss = 0

        with torch.no_grad():
            for input, target in tqdm(dl, desc=f"Validation", total=len(dl), disable=self.rank != 0):
                pred = self.model.forward(input)

                loss += tf.l1_loss(pred, target)

        if self.writer is not None:
            self.writer.add_scalar("Loss/val", loss.item() / len(dl), self.epoch)

            input[:, 0][input[:, 0] == 0] = 0.60
            input[:, 0][input[:, 0] == 1] = 0.83
            input[:, 0][input[:, 0] == 2] = 0.91
            input[:, 0][input[:, 0] == 3] = 0.97

            imgs = torch.cat([input, pred], dim=1)[:100]
            imgs = vlab_to_rgb(imgs)

            self.writer.add_images(f"Pred Images", imgs, self.epoch)

    def checkpoint(self):
        if self.rank != 0: return

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


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def load_dataset(dataset, rank, world_size):
    ds_memory = torch.empty(0, 3, 112, 128, dtype=torch.float32)

    paths = sorted([
        os.path.join(dataset, path)
        for path in os.listdir(dataset)
        if path.endswith(".npz")
    ])

    for i, path in tqdm(enumerate(paths), desc="Loading dataset", total=len(paths), disable=rank != 0):
        if i % world_size != rank:
            continue

        print(f"Loading {path} on rank {rank}")

        chunk = torch.tensor(np.load(path)["imgs"], dtype=torch.float32)

        grey = chunk[:, :1]
        rgb = chunk[:, 1:] / 255.0

        lab = (
            rgb_to_lab(rgb.movedim(0, 1).reshape(3, -1))
            .view(3, -1, 112, 128)
            .movedim(0, 1)
        )

        ds_memory = torch.cat([ds_memory, torch.cat([grey, lab[:, 1:]], dim=1)], dim=0)

    ds_memory = ds_memory.to(f"cuda:{rank}")

    return GBColorizeDataset(ds_memory)


def train_ddp(rank, world_size, model_name, dataset, epochs, batch_size, lr):
    setup_ddp(rank, world_size)
    device = f"cuda:{rank}"

    # Dataset
    ds = load_dataset(dataset, rank, world_size)
    train_ds, val_ds = random_split(ds, [0.95, 0.05])

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = MODELS[model_name]()
    model.init_weights()
    model.to(device)

    model = torch.compile(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train
    run_name = f"{model_name}_{batch_size}_{lr}_{epochs}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    trainer = Trainer(model, optim, run_name, device, rank)
    trainer.train(epochs, train_dl, val_dl)

    cleanup_ddp()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python train.py <model> <dataset> <epochs> <batch_size> <lr>")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset = sys.argv[2]
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    lr = float(sys.argv[5])

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_ddp,
        args=(world_size, model_name, dataset, epochs, batch_size, lr),
        nprocs=world_size,
        join=True,
    )

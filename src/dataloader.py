import os
import numpy as np

import torch
from torch.utils.data import Dataset

from torchtyping import TensorType

from utils.color import *

class GBColorizeDataset(Dataset):
    ds: TensorType["count", 4, 112, 128]
    device: str

    def __init__(self, ds: TensorType["count", 4, 112, 128], device: str):
        self.ds = ds
        self.device = device

    def __len__(self) -> int:
        return self.ds.shape[0]
    
    def __getitem__(self, idx) -> tuple[TensorType[1, 112, 128], TensorType[2, 112, 128]]:
        img = self.ds[idx].to(torch.float32).to(self.device, non_blocking=True)

        grey = img[:1]
        rgb = img[1:] / 255.0

        lab = vsingle_rgb_to_lab(rgb)
        
        return grey, lab[1:]

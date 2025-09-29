import os
import numpy as np

import torch
from torch.utils.data import Dataset

from torchtyping import TensorType

from utils.color import *

class GBColorizeDataset(Dataset):
    ds: TensorType["count", 3, 112, 128]
    
    def __init__(self, ds: TensorType["count", 3, 112, 128]):
        self.ds = ds

    def __len__(self) -> int:
        return self.ds.shape[0]
    
    def __getitem__(self, idx) -> tuple[TensorType[1, 112, 128], TensorType[2, 112, 128]]:
        return self.ds[idx, :1], self.ds[idx, 1:]

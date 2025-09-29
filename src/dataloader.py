import os
from typing import Iterator
import numpy as np

import torch
from torch.utils.data import IterableDataset

from torchtyping import TensorType

from utils.color import *

class GBColorizeDataset(IterableDataset):
    files: list[str]
    shuffle: bool
    
    def __init__(self, path: str, range: slice, shuffle: bool = True):
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")][range]
        self.shuffle = shuffle

    def __len__(self) -> int:
        return sum(np.load(file)["imgs"].shape[0] for file in self.files)

    def __iter__(self) -> Iterator[tuple[TensorType[1, 112, 128], TensorType[2, 112, 128]]]:
        for file in self.files:
            ds = torch.tensor(np.load(file)["imgs"], dtype=torch.uint8)

            if self.shuffle:
                ds = ds[torch.randperm(ds.shape[0])]

            yield from ds
            
            # for img in ds:
            #     img = img.to(torch.float32)

            #     grey = img[:1]
            #     rgb = img[1:] / 255.0

            #     lab = vrgb_to_lab(rgb.unsqueeze(0)).squeeze(0)

            #     yield grey, lab[1:]

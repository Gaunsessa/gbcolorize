import numpy as np

import torch
from torch.utils.data import Dataset

from torchtyping import TensorType

from utils.color import *

class GBColorizeDataset(Dataset):
    ds: TensorType["count", 4, 112, 128]
    
    def __init__(self, path: str):
        self.ds = torch.tensor(np.load(path)["imgs"], dtype=torch.uint8)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx) -> tuple[TensorType[1, 112, 128], TensorType[2, 112, 128]]:
        img = self.ds[idx].to(torch.float32)

        grey = img[:1]
        rgb = img[1:] / 255.0

        lab = vrgb_to_lab(rgb.unsqueeze(0)).squeeze(0)
        
        return grey, lab[1:]
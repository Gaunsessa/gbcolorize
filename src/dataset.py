import os
import sys
import glob
import numpy as np

import torch

from tqdm import tqdm

from torchtyping import TensorType

from utils.color import *
from utils.img import *


def process_imgs(img_paths: list[str]) -> TensorType["batch", 3, 112, 128]:
    imgs = torch.stack([scale_img(read_img(path)) for path in img_paths])

    imgs = vrgb_to_lab(imgs)

    # imgs = luma_dither(imgs)

    # gres = greyscale_idx_img(imgs)

    # imgs = vlab_to_rgb(imgs)
    # imgs = torch.round(imgs * 255)

    # imgs = torch.cat([gres, imgs], dim=1)

    return imgs.to(torch.float16)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset.py <input_dir> <output_path> <chunk_size>")
        sys.exit(1)

    imgs = glob.glob(os.path.join(sys.argv[1], "**", "*.jpg"), recursive=True)
    output_path = sys.argv[2]
    chunks_size = int(sys.argv[3])

    chunks = [imgs[i : i + chunks_size] for i in range(0, len(imgs), chunks_size)]

    for i, chunk in tqdm(enumerate(chunks), desc="Processing chunks", total=len(chunks)):
        chunk = process_imgs(chunk).detach().cpu().numpy().astype(np.float16)
        np.savez_compressed(os.path.join(output_path, f"{i}.npz"), imgs=chunk)

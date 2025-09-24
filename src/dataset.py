import os
import sys

import torch
import torchvision as tv
import torchvision.transforms.functional as vf

from tqdm import tqdm

from torchtyping import TensorType

from utils.color import *
from utils.img import *


def process_imgs(img_paths: list[str]) -> TensorType["batch", 3, 112, 128]:
    imgs = torch.stack([scale_img(read_img(path)) for path in img_paths])

    imgs = vrgb_to_lab(imgs)

    imgs = luma_dither(imgs)

    gres = greyscale_idx_img(imgs)

    imgs = vlab_to_rgb(imgs)
    imgs = torch.round(imgs * 255)

    imgs = torch.cat([gres, imgs], dim=1)

    return imgs.to(torch.uint8)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset.py <input_dir> <output_dir> <chunk_size>")
        sys.exit(1)

    imgs = os.listdir(sys.argv[1])
    output_dir = sys.argv[2]
    chunks_size = int(sys.argv[3])

    chunks = [imgs[i:i + chunks_size]
              for i in range(0, len(imgs), chunks_size)]

    print(f"Processing {len(chunks)} chunks into {output_dir}")

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        imgs = process_imgs([os.path.join(sys.argv[1], p) for p in chunk])

        torch.save(imgs, os.path.join(output_dir, f"{i}.pt"))

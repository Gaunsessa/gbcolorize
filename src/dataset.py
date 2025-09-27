import os
import sys
import glob
import h5py
import random
import numpy as np

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


def save_chunk(chunks: list[list[str]], output_path: str, chunks_size: int):
    length = sum(len(chunk) for chunk in chunks)

    print(f"Processing {len(chunks)} chunks into {output_path}")

    with h5py.File(output_path, "w") as f:
        ds = f.create_dataset(
            "images",
            shape=(length, 4, 112, 128),
            dtype="uint8",
            compression="gzip",
            chunks=(chunks_size, 4, 112, 128),
        )

        start = 0
        for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
            imgs = process_imgs(chunk)
            end = start + imgs.shape[0]
            ds[start:end] = imgs
            start = end


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset.py <input_dir> <output_path> <chunk_size>")
        sys.exit(1)

    imgs = glob.glob(os.path.join(sys.argv[1], "**", "*.jpg"), recursive=True)
    output_path = sys.argv[2]
    chunks_size = int(sys.argv[3])

    chunks = [imgs[i : i + chunks_size] for i in range(0, len(imgs), chunks_size)]
    chunks = [
        process_imgs(chunk).detach().cpu().numpy().astype(np.uint8)
        for chunk in tqdm(chunks, desc="Processing chunks")
    ]
    chunks = np.concatenate(chunks)

    np.savez_compressed(output_path, imgs=chunks)

    # random.shuffle(chunks)

    # n = len(chunks)
    # n_train = int(n * 0.7)
    # n_test = int(n * 0.2)
    # n_val = n - n_train - n_test

    # train_chunks = chunks[:n_train]
    # test_chunks = chunks[n_train:n_train + n_test]
    # val_chunks = chunks[n_train + n_test:]

    # print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}, Val chunks: {len(val_chunks)}")

    # save_chunk(train_chunks, f"{output_path}_train.h5", chunks_size)
    # save_chunk(test_chunks, f"{output_path}_test.h5", chunks_size)
    # save_chunk(val_chunks, f"{output_path}_val.h5", chunks_size)

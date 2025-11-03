import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils.color import get_color_bins, quantize_colors, vrgb_to_lab
from utils.img import read_img, scale_img


def process_imgs(img_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.stack([scale_img(read_img(path)) for path in img_paths])

    imgs = vrgb_to_lab(imgs)

    bins = get_color_bins()

    res = torch.empty(imgs.shape[0], 2, imgs.shape[2], imgs.shape[3])

    for i in range(0, imgs.shape[0], 500):
        res[i : i + 500] = quantize_colors(imgs[i : i + 500], bins)

    # imgs = luma_dither(imgs)

    # gres = greyscale_idx_img(imgs)

    # imgs = vlab_to_rgb(imgs)
    # imgs = torch.round(imgs * 255)

    # imgs = torch.cat([gres, imgs], dim=1)

    return res[:, :1].to(torch.float16), res[:, 1:].to(torch.uint8)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset.py <input_dir> <output_path> <chunk_size>")
        sys.exit(1)

    imgs = glob.glob(os.path.join(sys.argv[1], "**", "*.jpg"), recursive=True)
    output_path = sys.argv[2]
    chunks_size = int(sys.argv[3])

    chunks = [imgs[i : i + chunks_size] for i in range(0, len(imgs), chunks_size)]

    for i, chunk in tqdm(
        enumerate(chunks), desc="Processing chunks", total=len(chunks)
    ):
        luma, color = process_imgs(chunk)

        luma = luma.detach().cpu().numpy().astype(np.float16)
        color = color.detach().cpu().numpy().astype(np.uint8)

        np.savez_compressed(
            os.path.join(output_path, f"{i}.npz"), luma=luma, color=color
        )

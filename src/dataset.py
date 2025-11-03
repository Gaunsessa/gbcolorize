import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils.color import get_color_bins, quantize_colors, vrgb_to_lab
from utils.img import read_img, scale_img


def process_imgs(
    img_paths: list[str], bins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = torch.stack([scale_img(read_img(path)) for path in img_paths])

    imgs = vrgb_to_lab(imgs)

    res = torch.empty(imgs.shape[0], 2, imgs.shape[2], imgs.shape[3])

    for i in range(0, imgs.shape[0], 500):
        res[i : i + 500] = quantize_colors(imgs[i : i + 500], bins)

    batch_idx, batch_counts = res[:, 1].unique(return_counts=True)

    return (
        res[:, :1].to(torch.float16),
        res[:, 1:].to(torch.uint8),
        batch_idx,
        batch_counts,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset.py <input_dir> <output_path> <chunk_size>")
        sys.exit(1)

    imgs = glob.glob(os.path.join(sys.argv[1], "**", "*.jpg"), recursive=True)
    output_path = sys.argv[2]
    chunks_size = int(sys.argv[3])

    chunks = [imgs[i : i + chunks_size] for i in range(0, len(imgs), chunks_size)]

    bins = get_color_bins()
    bin_counts = torch.zeros(bins.shape[0])

    for i, chunk in tqdm(
        enumerate(chunks), desc="Processing chunks", total=len(chunks)
    ):
        luma, color, batch_idx, batch_counts = process_imgs(chunk, bins)

        bin_counts[batch_idx.to(torch.int)] += batch_counts / batch_counts.sum()

        luma = luma.detach().cpu().numpy().astype(np.float16)
        color = color.detach().cpu().numpy().astype(np.uint8)

        np.savez_compressed(
            os.path.join(output_path, f"{i}.npz"), luma=luma, color=color
        )

    bin_weights = 1 - bin_counts / len(chunks)
    torch.save(bin_weights, os.path.join(output_path, "bin_weights.pt"))

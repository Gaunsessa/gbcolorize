from concurrent.futures import ThreadPoolExecutor
import os
import glob
import argparse

import torch
from tqdm import tqdm

from torchvision.io import write_jpeg

from utils.color import get_color_bins, quantize_colors, vrgb_to_lab
from utils.img import read_img, scale_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)

    args = parser.parse_args()

    imgs = glob.glob(os.path.join(args.input, "**", "*.jpg"), recursive=True)

    os.makedirs(os.path.join(args.output, "imgs"), exist_ok=True)

    bins = get_color_bins()
    bin_counts = torch.zeros(bins.shape[0], dtype=torch.float64)

    chunks = [
        zip(range(i, i + args.chunk_size), imgs[i : i + args.chunk_size])
        for i in range(0, len(imgs), args.chunk_size)
    ]

    def process_chunk(chunk):
        indicies, paths = zip(*chunk)
        imgs = torch.stack([scale_img(read_img(path)) for path in paths])

        labs = vrgb_to_lab(imgs)
        labs = quantize_colors(labs, bins)

        batch_idx, batch_counts = labs[:, 1].unique(return_counts=True)
        bin_counts[batch_idx.to(torch.int)] += batch_counts

        imgs = (imgs * 255).to(torch.uint8)

        for idx, img in zip(indicies, imgs):
            write_jpeg(img, os.path.join(args.output, f"imgs/{idx}.jpg"), quality=100)
            
        return len(imgs)

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        with tqdm(total=len(imgs)) as pbar:
            for count in exe.map(process_chunk, chunks):
                pbar.update(count)

    bin_weights = 1 - bin_counts / len(imgs)
    torch.save(bin_weights, os.path.join(args.output, "bin_weights.pt"))

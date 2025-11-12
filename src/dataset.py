import os
import glob
import torch
import shutil
import tarfile
import argparse
import random

import numpy as np

from tqdm import tqdm

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils.img import read_img, scale_img
from utils.color import get_color_bins, quantize_colors, vrgb_to_lab


def process_chunk(
    chunk: list[tuple[int, str]], bins: torch.Tensor, device: str
) -> tuple[int, list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    indicies, paths = zip(*chunk)
    imgs = torch.stack([scale_img(read_img(path)) for path in paths]).to(device)

    labs = vrgb_to_lab(imgs)
    labs = quantize_colors(labs, bins)

    bin_counts = torch.zeros(bins.shape[0], dtype=torch.float64).to(device)
    batch_idx, batch_counts = labs[:, 1].unique(return_counts=True)
    bin_counts[batch_idx.to(torch.int)] += batch_counts

    lumas = labs[:, :1].to(torch.float16)
    colors = labs[:, 1:].to(torch.uint8)

    return len(imgs), list(indicies), bin_counts, lumas, colors


def process_shard(
    imgs: list[tuple[int, str]],
    bins: torch.Tensor,
    device: str,
    output: str,
    chunk_size: int,
    workers: int,
) -> torch.Tensor:
    chunks = [imgs[i : i + chunk_size] for i in range(0, len(imgs), chunk_size)]

    bin_counts = torch.zeros(bins.shape[0], dtype=torch.float64).to(device)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        proc_fn = partial(process_chunk, bins=bins, device=device)

        with tqdm(total=len(imgs)) as pbar:
            for count, indicies, batch_bin_counts, lumas, colors in exe.map(
                proc_fn, chunks
            ):
                bin_counts += batch_bin_counts

                for idx, luma, color in zip(indicies, lumas, colors):
                    np.savez_compressed(
                        os.path.join(output, f"{idx:07d}.luma.npz"),
                        luma=luma.cpu().numpy(),
                    )
                    np.savez_compressed(
                        os.path.join(output, f"{idx:07d}.color.npz"),
                        color=color.cpu().numpy(),
                    )

                pbar.update(count)

    return bin_counts


def process_split(
    samples: list[tuple[int, str]],
    prefix: str,
    shard_size: int,
    chunk_size: int,
    workers: int,
    bins: torch.Tensor,
    device: str,
    tmp_dir: str,
    data_dir: str,
) -> torch.Tensor:
    shards = [samples[i : i + shard_size] for i in range(0, len(samples), shard_size)]

    bin_counts = torch.zeros(bins.shape[0], dtype=torch.float64).to(device)

    for shard_idx, shard in enumerate(shards):
        bin_counts += process_shard(
            shard,
            bins,
            device,
            tmp_dir,
            chunk_size,
            workers,
        )

        with tarfile.open(
            os.path.join(data_dir, f"{prefix}_{shard_idx:07d}.tar"), "w"
        ) as tar:
            for idx, _ in shard:
                tar.add(
                    os.path.join(tmp_dir, f"{idx:07d}.luma.npz"),
                    arcname=f"{idx:07d}.luma.npz",
                )
                tar.add(
                    os.path.join(tmp_dir, f"{idx:07d}.color.npz"),
                    arcname=f"{idx:07d}.color.npz",
                )

                os.remove(os.path.join(tmp_dir, f"{idx:07d}.luma.npz"))
                os.remove(os.path.join(tmp_dir, f"{idx:07d}.color.npz"))

    torch.save(len(samples), os.path.join(data_dir, f"{prefix}_length.pt"))

    return bin_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split_size", type=float, required=True)
    parser.add_argument("--shard_size", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)

    args = parser.parse_args()

    # Load images
    imgs = glob.glob(os.path.join(args.input, "**", "*.jpg"), recursive=True)
    random.shuffle(imgs)

    # Generate bins
    bins = get_color_bins().to(args.device)

    # Create temporary and data directories
    tmp_dir = os.path.join(args.output, "tmp")
    data_dir = os.path.join(args.output, "data")

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Split into train and val
    train_size = int(len(imgs) * args.split_size)
    train_samples = list(enumerate(imgs[:train_size]))
    val_samples = list(enumerate(imgs[train_size:]))

    # Process splits
    bin_counts = process_split(
        train_samples,
        "train",
        args.shard_size,
        args.chunk_size,
        args.workers,
        bins,
        args.device,
        tmp_dir,
        data_dir,
    )

    process_split(
        val_samples,
        "val",
        args.shard_size,
        args.chunk_size,
        args.workers,
        bins,
        args.device,
        tmp_dir,
        data_dir,
    )

    # Remove temporary directory
    shutil.rmtree(tmp_dir)

    # Save bin weights
    bin_weights = 1 - bin_counts / len(train_samples)
    torch.save(bin_weights.cpu(), os.path.join(args.output, "bin_weights.pt"))

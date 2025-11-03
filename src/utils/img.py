import torch
import torchvision as tv
import torchvision.transforms.functional as vf

from utils.color import vlab_to_rgb


def batch_display_img(imgs: torch.Tensor) -> torch.Tensor:
    """imgs: RGB"""

    return tv.utils.make_grid(imgs, nrow=int(imgs.shape[0] ** 0.5))


def tensor_to_pil(img: torch.Tensor, scale: int = 3):
    """img: RGB"""

    # Prevent overflow
    img = torch.clamp(img.clone(), 0, 1)

    # Resize to scale
    img = vf.resize(
        img.clone(),
        [img.shape[1] * scale, img.shape[2] * scale],
        tv.transforms.InterpolationMode.NEAREST,
    )

    return vf.to_pil_image(img)


def read_img(path: str) -> torch.Tensor:
    img = tv.io.read_image(path, mode=tv.io.ImageReadMode.RGB)

    # Convert to float
    img = (img / 255.0).to(torch.float32)

    return img


def scale_img(img: torch.Tensor) -> torch.Tensor:
    # Square image
    sdim = min(img.shape[1:])
    img = vf.center_crop(img, [int(sdim * (112 / 128)), sdim])

    # Resize
    img = vf.resize(img, [112, 128])

    return img


def luma_dither(img: torch.Tensor) -> torch.Tensor:
    """img: LAB"""

    # Clone
    img = img.clone()

    dither = (
        torch.tensor(
            [
                [0.0, 12.0, 3.0, 15.0],
                [8.0, 4.0, 11.0, 7.0],
                [2.0, 14.0, 1.0, 13.0],
                [10.0, 6.0, 9.0, 5.0],
            ]
        )
        / 16
        - 0.5
    )

    dither /= 20

    dither = dither.repeat(28, 32)

    # Dither
    img[:, 0] += dither

    # Quantize
    img[:, 0] /= img[:, 0].amax(dim=(1, 2), keepdim=True)

    img[:, 0][img[:, 0] <= 0.83] = 0.60
    img[:, 0][(img[:, 0] > 0.83) & (img[:, 0] <= 0.91)] = 0.83
    img[:, 0][(img[:, 0] > 0.91) & (img[:, 0] <= 0.97)] = 0.91
    img[:, 0][img[:, 0] > 0.97] = 0.97

    return img


def greyscale_idx_img(img: torch.Tensor) -> torch.Tensor:
    """img: LAB"""

    # Clone
    img = img.clone()

    # Set other channels to zero
    img[:, 1:] = 0

    # Convert back to rgb
    img = vlab_to_rgb(img)
    img = torch.round(img * 255)[:, :1]

    img[img == 10] = 0
    img[img == 73] = 1
    img[img == 135] = 2
    img[img == 207] = 3

    return img

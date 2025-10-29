import torch


# Color transformations
def nonlinear_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0.0031308, (1.055 * x ** (1.0 / 2.4) - 0.055), (12.92 * x))


def linear_to_nonlinear(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)

    return torch.where(x >= 0.04045, ((x + 0.055) / 1.055).pow(2.4), x / 12.92)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    # This operation clones
    rgb = nonlinear_to_linear(rgb)

    l = 0.4122214708 * rgb[0] + 0.5363325363 * rgb[1] + 0.0514459929 * rgb[2]
    m = 0.2119034982 * rgb[0] + 0.6806995451 * rgb[1] + 0.1073969566 * rgb[2]
    s = 0.0883024619 * rgb[0] + 0.2817188376 * rgb[1] + 0.6299787005 * rgb[2]

    l_ = l ** (1 / 3)
    m_ = m ** (1 / 3)
    s_ = s ** (1 / 3)

    res = torch.empty_like(rgb)
    res[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    res[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    res[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    # c = torch.sqrt(res[1] ** 2 + res[2] ** 2)
    # h = torch.atan2(res[2], res[1])

    # res[1] = c
    # res[2] = h

    return res


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    # a = lab[1] * torch.cos(lab[2])
    # b = lab[1] * torch.sin(lab[2])
    a = lab[1]
    b = lab[2]

    l_ = lab[0] + 0.3963377774 * a + 0.2158037573 * b
    m_ = lab[0] - 0.1055613458 * a - 0.0638541728 * b
    s_ = lab[0] - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    res = torch.empty_like(lab)
    res[0] = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    res[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    res[2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return linear_to_nonlinear(res)


def get_color_bins(steps=50, quant_lum=0.923632) -> torch.Tensor:
    a_grid, b_grid = torch.meshgrid(
        torch.linspace(-0.3, 0.3, steps), torch.linspace(-0.3, 0.3, steps)
    )

    lab_grid = torch.stack([torch.full_like(a_grid, quant_lum), a_grid, b_grid], dim=-1)
    lab_grid = lab_grid.movedim(-1, 0).view(3, -1)

    rgb_grid = lab_to_rgb(lab_grid)

    lab_grid = lab_grid[:, ~(rgb_grid > 1).any(dim=0)]

    return lab_grid[1:].movedim(0, -1)


def quantize_colors(img: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    ab_flat = torch.stack([img[:, 1].flatten(), img[:, 2].flatten()], dim=1)
    dists = torch.cdist(ab_flat, bins)

    closest_idxs = torch.argmin(dists, dim=1, keepdim=True).view(
        img.shape[0], img.shape[2], img.shape[3]
    )

    return torch.stack([img[:, 0], closest_idxs], dim=1)


def dequantize_colors(img: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return torch.cat([img[:, :1], bins[img[:, 1].to(torch.int)].movedim(-1, 1)], dim=1)


# This double vmap is horrific
vsingle_rgb_to_lab = torch.vmap(rgb_to_lab, in_dims=1, out_dims=1)

vrgb_to_lab = torch.vmap(torch.vmap(rgb_to_lab, in_dims=1, out_dims=1))
vlab_to_rgb = torch.vmap(torch.vmap(lab_to_rgb, in_dims=1, out_dims=1))

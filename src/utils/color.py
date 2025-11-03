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


def get_lab_ab_bounds(quant_lum: float) -> tuple[float, float, float, float]:
    all_rgb = (
        torch.stack(
            torch.meshgrid(
                torch.arange(256, dtype=torch.uint8),
                torch.arange(256, dtype=torch.uint8),
                torch.arange(256, dtype=torch.uint8),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .movedim(-1, 0)
        / 255.0
    )

    all_lab = rgb_to_lab(all_rgb)
    all_lab = all_lab[:, torch.abs(all_lab[0] - quant_lum) < 0.0001]

    return (
        all_lab[1].min().item(),
        all_lab[1].max().item(),
        all_lab[2].min().item(),
        all_lab[2].max().item(),
    )


def diverse_k_center_clustering(
    points: torch.Tensor, num_centers: int, start: int
) -> torch.Tensor:
    num_points, dim = points.shape

    centers = torch.empty((num_centers, dim), device=points.device)
    centers[0] = points[start]

    min_dists = torch.full((num_points,), float("inf"), device=points.device)
    for i in range(1, num_centers):
        dists = ((points - centers[i - 1 : i]) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        centers[i] = points[min_dists.argmax()]

    return centers


def get_color_bins(steps=1000, quant_lum=0.97, count=256) -> torch.Tensor:
    a_min, a_max, b_min, b_max = get_lab_ab_bounds(quant_lum)

    a_grid, b_grid = torch.meshgrid(
        torch.linspace(a_min, a_max, steps),
        torch.linspace(b_min, b_max, steps),
        indexing="ij",
    )

    lab_grid = torch.stack([torch.full_like(a_grid, quant_lum), a_grid, b_grid], dim=-1)
    lab_grid = lab_grid.movedim(-1, 0).view(3, -1)

    rgb_grid = lab_to_rgb(lab_grid)

    lab_grid = lab_grid[:, (rgb_grid > 1).sum(dim=0) <= 1]

    bins = lab_grid[1:].movedim(0, -1)
    bins = torch.cat([torch.zeros((1, bins.shape[1])), bins], dim=0)

    bins = diverse_k_center_clustering(bins, count, 0)
    # bins = kmeans_clustering(bins, count)

    return bins


def precompute_color_bins_weights(bins: torch.Tensor, sigma: float = 0.05, k: int = 5):
    x2 = (bins**2).sum(-1, keepdim=True)
    y2 = (bins**2).sum(-1)
    xy = bins @ bins.T
    dist2 = x2 + y2 - 2 * xy

    dist2 = dist2 / dist2.max(dim=-1, keepdim=True)[0]

    knn_dists, knn_idx = torch.topk(dist2, k=k, largest=False)

    knn_weights = torch.exp(-knn_dists / (2 * sigma**2))

    return knn_idx, knn_weights


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

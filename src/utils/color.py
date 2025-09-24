import torch

# Color transformations
def nonlinear_to_linear(x: float) -> float:
   # This has to be done to allow for vmap
   return (x >= 0.0031308) * (1.055 * x ** (1.0 / 2.4) - 0.055) + (x < 0.0031308) * (12.92 * x)

def linear_to_nonlinear(x: float) -> float:
   # This has to be done to allow for vmap
   return (x >= 0.04045) * (((x + 0.055) / (1 + 0.055)) ** 2.4) + (x < 0.04045) * (x / 12.92)

def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
   # This operation clones
   rgb = nonlinear_to_linear(rgb)

   l = 0.4122214708 * rgb[0] + 0.5363325363 * rgb[1] + 0.0514459929 * rgb[2]
   m = 0.2119034982 * rgb[0] + 0.6806995451 * rgb[1] + 0.1073969566 * rgb[2]
   s = 0.0883024619 * rgb[0] + 0.2817188376 * rgb[1] + 0.6299787005 * rgb[2]

   l_ = l ** (1 / 3)
   m_ = m ** (1 / 3)
   s_ = s ** (1 / 3)

   rgb[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
   rgb[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
   rgb[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

   c = torch.sqrt(rgb[1] ** 2 + rgb[2] ** 2)
   h = torch.atan2(rgb[2], rgb[1])

   rgb[1] = c
   rgb[2] = h

   return rgb

def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
   lab = lab.clone()

   a = lab[1] * torch.cos(lab[2])
   b = lab[1] * torch.sin(lab[2])

   lab[1] = a
   lab[2] = b

   l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2]
   m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2]
   s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2]

   l = l_ ** 3
   m = m_ ** 3
   s = s_ ** 3

   lab[0] = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
   lab[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
   lab[2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

   return linear_to_nonlinear(lab)

# This double vmap is horrific
vrgb_to_lab = torch.vmap(torch.vmap(rgb_to_lab, in_dims = 1, out_dims = 1))
vlab_to_rgb = torch.vmap(torch.vmap(lab_to_rgb, in_dims = 1, out_dims = 1))
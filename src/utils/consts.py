from models.unet import UNet
from models.conv import GBConvModel
from models.resp import RespModel
from models.efficient import EfficientModel

MODELS = {
    "unet": UNet,
    "conv": GBConvModel,
    "resp": RespModel,
    "eff": EfficientModel,
}
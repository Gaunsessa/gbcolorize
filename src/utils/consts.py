from models.unet import UNet
from models.conv import GBConvModel
from models.resp import RespModel
from models.efficient import EfficientModel
from models.efficient2 import Efficient2Model

MODELS = {
    "unet": UNet,
    "conv": GBConvModel,
    "resp": RespModel,
    "eff": EfficientModel,
    "eff2": Efficient2Model
}
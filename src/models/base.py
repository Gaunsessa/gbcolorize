import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def freeze_encoder(self, freeze: bool = True): ...

    @abstractmethod
    def init_weights(self): ...

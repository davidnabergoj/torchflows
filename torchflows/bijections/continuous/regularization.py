import torch
import torch.nn as nn


class GeodesicRegularization(nn.Module):
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self.register_buffer('coef', torch.tensor(coef))
        self.register_buffer('value', torch.tensor(0.))


class JacobianRegularization(nn.Module):
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self.register_buffer('coef', torch.tensor(coef))
        self.register_buffer('value', torch.tensor(0.))

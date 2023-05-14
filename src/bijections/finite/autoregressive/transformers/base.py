import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        raise NotImplementedError

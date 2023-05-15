from typing import Tuple

import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer


class Affine(Transformer):
    def __init__(self, scale_transform: callable = torch.exp):
        super().__init__()
        self.scale_transform = scale_transform

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.scale_transform(h[..., 0])
        if self.scale_transform is torch.exp:
            log_alpha = h[..., 0]
        else:
            log_alpha = torch.log(alpha)
        beta = h[..., 1]
        log_det = torch.sum(log_alpha, dim=-1)
        return alpha * x + beta, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.scale_transform(h[..., 0])
        if self.scale_transform is torch.exp:
            log_alpha = h[..., 0]
        else:
            log_alpha = torch.log(alpha)
        beta = h[..., 1]
        log_det = -torch.sum(log_alpha, dim=-1)
        return (z - beta) / alpha, log_det

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


class InverseAffine(Transformer):
    """
    Inverse affine transformer, numerically stable for data generation.
    """

    def __init__(self, scale_transform: callable = torch.exp):
        super().__init__()
        self.affine = Affine(scale_transform=scale_transform)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.affine.inverse(x, h)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.affine.forward(z, h)


class Shift(Transformer):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        log_det = torch.zeros((x.shape[0],), device=x.device)
        return x + beta, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        log_det = torch.zeros((z.shape[0],), device=z.device)
        return z - beta, log_det

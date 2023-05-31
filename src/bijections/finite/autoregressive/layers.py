import math
from typing import Tuple

import torch
import torch.nn as nn

from src.bijections.finite.autoregressive.conditioner_transforms import MADE, FeedForward, Linear
from src.bijections.finite.autoregressive.conditioners.coupling import Coupling
from src.bijections.finite.autoregressive.conditioners.masked import MaskedAutoregressive
from src.bijections.finite.autoregressive.transformers.affine import Affine, Shift
from src.bijections.finite.autoregressive.transformers.base import Transformer
from src.bijections.finite.autoregressive.transformers.spline import RationalQuadraticSpline
from src.bijections.finite.base import Bijection


class CouplingBijection(Bijection):
    def __init__(self, n_dim, constant_dims, constants, conditioner_transform: nn.Module, transformer: Transformer):
        super().__init__()
        self.transformer = transformer
        self.conditioner = Coupling(
            transform=conditioner_transform,
            constants=constants,
            constant_dims=constant_dims,
            n_dim=n_dim
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(x)
        z, log_det = self.transformer(x, h)
        return z, log_det

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(z)
        x, log_det = self.transformer.inverse(z, h)
        return x, log_det


class AffineCoupling(CouplingBijection):
    def __init__(self, n_dim, constant_dims, conditioner_transform: nn.Module, scale_transform: callable = torch.exp):
        default_log_scale = 0.0
        default_shift = 0.0
        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            constants=torch.tensor([default_log_scale, default_shift]),
            conditioner_transform=conditioner_transform,
            transformer=Affine(scale_transform=scale_transform)
        )


class ShiftCoupling(CouplingBijection):
    def __init__(self, n_dim, constant_dims, conditioner_transform: nn.Module, **kwargs):
        default_shift = 0.0
        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            constants=torch.tensor([default_shift]),
            conditioner_transform=conditioner_transform,
            transformer=Shift()
        )


class RationalQuadraticSplineCoupling(CouplingBijection):
    def __init__(self,
                 n_dim,
                 n_bins: int,
                 constant_dims,
                 conditioner_transform: nn.Module,
                 **kwargs):
        assert n_bins >= 2
        default_unconstrained_widths = torch.zeros(n_bins)
        default_unconstrained_heights = torch.zeros(n_bins)
        default_unconstrained_derivatives = torch.full(size=(n_bins - 1,), fill_value=math.log(math.expm1(1)))
        constants = torch.cat([
            default_unconstrained_widths,
            default_unconstrained_heights,
            default_unconstrained_derivatives
        ])
        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            constants=constants,
            conditioner_transform=conditioner_transform,
            transformer=RationalQuadraticSpline(n_bins=n_bins, **kwargs)
        )


class LinearAffineCoupling(AffineCoupling):
    def __init__(self, n_dim: int, **kwargs):
        assert n_dim >= 2

        # Set up the input and output dimensions
        n_transformer_parameters = 2
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the linear conditioner transform
        lin_cond = Linear(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters
        )

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=lin_cond,
            **kwargs
        )


class LinearRationalQuadraticSplineCoupling(RationalQuadraticSplineCoupling):
    def __init__(self, n_dim, n_bins: int = 8, **kwargs):
        # Set up the input and output dimensions
        n_transformer_parameters = 3 * n_bins - 1
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the linear conditioner transform
        lin_cond = Linear(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters
        )
        super().__init__(
            n_dim=n_dim,
            n_bins=n_bins,
            constant_dims=constant_dims,
            conditioner_transform=lin_cond,
            **kwargs
        )


class LinearShiftCoupling(ShiftCoupling):
    def __init__(self, n_dim: int, **kwargs):
        assert n_dim >= 2

        # Set up the input and output dimensions
        n_transformer_parameters = 1
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the linear conditioner transform
        lin_cond = Linear(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters
        )

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=lin_cond,
            **kwargs
        )


class FeedForwardAffineCoupling(AffineCoupling):
    def __init__(self, n_dim: int, **kwargs):
        assert n_dim >= 2

        # Set up the input and output dimensions
        n_transformer_parameters = 2
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the neural network conditioner transform
        network = FeedForward(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters,
            **kwargs
        )

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=network,
            **kwargs
        )


class FeedForwardRationalQuadraticSplineCoupling(RationalQuadraticSplineCoupling):
    def __init__(self, n_dim, n_bins: int = 8, **kwargs):
        # Set up the input and output dimensions
        n_transformer_parameters = 3 * n_bins - 1
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the linear conditioner transform
        network = FeedForward(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters,
            **kwargs
        )

        super().__init__(
            n_dim=n_dim,
            n_bins=n_bins,
            constant_dims=constant_dims,
            conditioner_transform=network,
            **kwargs
        )


class FeedForwardShiftCoupling(ShiftCoupling):
    def __init__(self, n_dim: int, **kwargs):
        assert n_dim >= 2

        # Set up the input and output dimensions
        n_transformer_parameters = 1
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the neural network conditioner transform
        network = FeedForward(
            n_input_dims=n_constant_dims,
            n_output_dims=n_transformed_dims,
            n_output_parameters=n_transformer_parameters,
            **kwargs
        )

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=network,
            **kwargs
        )


class ForwardMaskedAutoregressive(Bijection):
    def __init__(self, n_dim: int, conditioner_transform: nn.Module, transformer: Transformer):
        super().__init__()
        self.transformer = transformer
        self.conditioner = MaskedAutoregressive(transform=conditioner_transform, n_dim=n_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(x)
        z, log_det = self.transformer(x, h)
        return z, log_det

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(size=(z.shape[0],), device=z.device)
        x = torch.clone(z)
        for i in torch.arange(z.shape[-1]):  # FIXME this probably messes up autograd b/c it overwrites the gradient?
            h = self.conditioner(torch.clone(x))
            tmp, log_det = self.transformer.inverse(x, h)
            x[:, i] = tmp[:, i]
        return x, log_det


class InverseMaskedAutoregressive(Bijection):
    def __init__(self, n_dim: int, conditioner_transform: nn.Module, transformer: Transformer):
        super().__init__()
        self.forward_masked_autoregressive = ForwardMaskedAutoregressive(n_dim, conditioner_transform, transformer)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_masked_autoregressive.inverse(x)

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_masked_autoregressive.forward(z)


class AffineForwardMaskedAutoregressive(ForwardMaskedAutoregressive):
    def __init__(self, n_dim: int, scale_transform: callable = torch.exp, **kwargs):
        transformer = Affine(scale_transform=scale_transform)
        conditioner_transform = MADE(
            n_input_dims=n_dim,
            n_output_dims=n_dim,
            n_output_parameters=2,
            **kwargs)
        super().__init__(n_dim, conditioner_transform, transformer)


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressive):
    def __init__(self, n_dim: int, scale_transform: callable = torch.exp, **kwargs):
        transformer = Affine(scale_transform=scale_transform)
        conditioner_transform = MADE(
            n_input_dims=n_dim,
            n_output_dims=n_dim,
            n_output_parameters=2,
            **kwargs)
        super().__init__(n_dim, conditioner_transform, transformer)

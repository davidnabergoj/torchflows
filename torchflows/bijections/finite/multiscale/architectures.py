from typing import Union, Tuple

import torch

from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Shift, Affine
from torchflows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from torchflows.bijections.finite.autoregressive.transformers.spline.linear import Linear as LinearRational
from torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid import (
    DeepSigmoid,
    DeepDenseSigmoid,
    DenseSigmoid
)
from torchflows.bijections.finite.multiscale.base import MultiscaleBijection, GlowCheckerboardCoupling, \
    GlowChannelWiseCoupling


def check_image_shape_for_multiscale_flow(event_shape, n_layers):
    if len(event_shape) != 3:
        raise ValueError("Multichannel image transformation are only possible for inputs with 3 axes.")
    if event_shape[1] % 2 != 0 or event_shape[2] % 2 != 0:
        raise ValueError("Image height and width must be divisible by 2.")
    if n_layers is not None and n_layers < 1:
        raise ValueError("Need at least one layer for multiscale flow.")

    # Check image height and width
    if n_layers is not None:
        if event_shape[1] % (2 ** n_layers) != 0:
            raise ValueError("Image height must be divisible by pow(2, n_layers).")
        elif event_shape[2] % (2 ** n_layers) != 0:
            raise ValueError("Image width must be divisible by pow(2, n_layers).")


def automatically_determine_n_layers(event_shape):
    if event_shape[1] % (2 ** 3) == 0 and event_shape[2] % (2 ** 3) == 0:
        # Try using 3 layers
        n_layers = 3
    elif event_shape[1] % (2 ** 2) == 0 and event_shape[2] % (2 ** 2) == 0:
        # Try using 2 layers
        n_layers = 2
    elif event_shape[1] % 2 == 0 and event_shape[2] % 2 == 0:
        n_layers = 1
    else:
        raise ValueError("Image height and width must be divisible by 2.")
    return n_layers


class MultiscaleRealNVP(MultiscaleBijection):
    """Multiscale version of Real NVP.

    Reference: Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self,
                 event_shape: Union[int, torch.Size, Tuple[int, ...]],
                 n_layers: int = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=Affine,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleNICE(MultiscaleBijection):
    """Multiscale version of NICE.

    References:
        - Dinh et al. "NICE: Non-linear Independent Components Estimation" (2015); https://arxiv.org/abs/1410.8516.
        - Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=Shift,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleRQNSF(MultiscaleBijection):
    """Multiscale version of C-RQNSF.

    References:
        - Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
        - Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=RationalQuadratic,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleLRSNSF(MultiscaleBijection):
    """Multiscale version of C-LRS.

    References:
        - Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
        - Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=LinearRational,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleDeepSigmoid(MultiscaleBijection):
    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=DeepSigmoid,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleDeepDenseSigmoid(MultiscaleBijection):
    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=DeepDenseSigmoid,
            n_blocks=n_layers,
            **kwargs
        )


class MultiscaleDenseSigmoid(MultiscaleBijection):
    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=DenseSigmoid,
            n_blocks=n_layers,
            **kwargs
        )


class AffineGlow(MultiscaleBijection):
    def __init__(self, event_shape: Union[int, torch.Size, Tuple[int, ...]], n_layers: int = None, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        if n_layers is None:
            n_layers = automatically_determine_n_layers(event_shape)
        check_image_shape_for_multiscale_flow(event_shape, n_layers)
        super().__init__(
            event_shape=event_shape,
            transformer_class=Affine,
            checkerboard_class=GlowCheckerboardCoupling,
            channel_wise_class=GlowChannelWiseCoupling,
            n_blocks=n_layers,
            **kwargs
        )

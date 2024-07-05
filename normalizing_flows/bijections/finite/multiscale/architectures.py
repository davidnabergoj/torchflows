import torch

from normalizing_flows.bijections.finite.autoregressive.layers import ElementwiseAffine
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.affine import Affine, Shift
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear import Linear as LinearRational
from normalizing_flows.bijections import BijectiveComposition
from normalizing_flows.bijections.finite.multiscale.base import MultiscaleBijection, FactoredBijection
import math


def make_factored_image_layers(event_shape,
                               transformer_class,
                               n_layers: int = 2):
    """
    Creates a list of image transformations consisting of coupling layers and squeeze layers.
    After each coupling, squeeze, coupling mapping, half of the channels are kept as is (not transformed anymore).

    :param event_shape: (c, 2^n, 2^m).
    :param transformer_class:
    :param n_layers:
    :return:
    """
    if len(event_shape) != 3:
        raise ValueError("Multichannel image transformation are only possible for inputs with three axes.")
    if bin(event_shape[1]).count("1") != 1:
        raise ValueError("Image height must be a power of two.")
    if bin(event_shape[2]).count("1") != 1:
        raise ValueError("Image width must be a power of two.")
    if n_layers < 1:
        raise ValueError

    log_height = math.log2(event_shape[1])
    log_width = math.log2(event_shape[2])
    if n_layers > min(log_height, log_width):
        raise ValueError("Too many layers for input image size")

    def recursive_layer_builder(event_shape_, n_layers_):
        msb = MultiscaleBijection(
            input_event_shape=event_shape_,
            transformer_class=transformer_class
        )
        if n_layers_ == 1:
            return msb

        c, h, w = msb.transformed_shape  # c is a multiple of 4 after squeezing

        small_bijection_shape = (c // 2, h, w)
        small_bijection_mask = (torch.arange(c) >= c // 2)[:, None, None].repeat(1, h, w)
        fb = FactoredBijection(
            event_shape=(c, h, w),
            small_bijection=recursive_layer_builder(
                event_shape_=small_bijection_shape,
                n_layers_=n_layers_ - 1
            ),
            small_bijection_mask=small_bijection_mask
        )
        composition = BijectiveComposition(
            event_shape=msb.event_shape,
            layers=[msb, fb]
        )
        composition.transformed_shape = fb.transformed_shape
        return composition

    bijections = [ElementwiseAffine(event_shape=event_shape)]
    bijections.append(recursive_layer_builder(bijections[-1].transformed_shape, n_layers))
    bijections.append(ElementwiseAffine(event_shape=bijections[-1].transformed_shape))
    return bijections


def make_image_layers_non_factored(event_shape,
                                   transformer_class,
                                   n_layers: int = 2):
    """
    Returns a list of bijections for transformations of images with multiple channels.
    """
    if len(event_shape) != 3:
        raise ValueError("Multichannel image transformation are only possible for inputs with three axes.")

    assert n_layers >= 1

    # TODO check that image shape is big enough for this number of layers (divisibility by 2)

    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers - 1):
        bijections.append(
            MultiscaleBijection(
                input_event_shape=bijections[-1].transformed_shape,
                transformer_class=transformer_class
            )
        )
    bijections.append(
        MultiscaleBijection(
            input_event_shape=bijections[-1].transformed_shape,
            transformer_class=transformer_class,
            n_checkerboard_layers=4,
            squeeze_layer=False,
            n_channel_wise_layers=0
        )
    )
    bijections.append(ElementwiseAffine(event_shape=bijections[-1].transformed_shape))
    return bijections


def make_image_layers(*args, factored: bool = False, **kwargs):
    if factored:
        return make_factored_image_layers(*args, **kwargs)
    else:
        return make_image_layers_non_factored(*args, **kwargs)


class MultiscaleRealNVP(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 3,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, Affine, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleNICE(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 3,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, Shift, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleRQNSF(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 3,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, RationalQuadratic, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleLRSNSF(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 3,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, LinearRational, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape

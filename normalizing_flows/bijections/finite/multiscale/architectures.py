import torch

from normalizing_flows.bijections.finite.autoregressive.layers import ElementwiseAffine
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.affine import Affine, Shift
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear import Linear as LinearRational
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid import (
    DeepSigmoid,
    DeepDenseSigmoid,
    DenseSigmoid
)
from normalizing_flows.bijections import BijectiveComposition
from normalizing_flows.bijections.finite.multiscale.base import MultiscaleBijection, FactoredBijection


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


def make_factored_image_layers(event_shape,
                               transformer_class,
                               n_layers: int = None):
    """
    Creates a list of image transformations consisting of coupling layers and squeeze layers.
    After each coupling, squeeze, coupling mapping, half of the channels are kept as is (not transformed anymore).

    :param event_shape: (c, 2^n, 2^m).
    :param transformer_class:
    :param n_layers:
    :return:
    """
    check_image_shape_for_multiscale_flow(event_shape, n_layers)
    if n_layers is None:
        n_layers = automatically_determine_n_layers(event_shape)
    check_image_shape_for_multiscale_flow(event_shape, n_layers)

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
                                   n_layers: int = None):
    """
    Returns a list of bijections for transformations of images with multiple channels.

    Let n be the number of layers. This sequence of bijections takes as input an image with shape (c, h, w) and outputs
    an image with shape (4 ** n * c, h / 2 ** n, w / 2 ** n). We require h and w to be divisible by 2 ** n.
    """
    check_image_shape_for_multiscale_flow(event_shape, n_layers)
    if n_layers is None:
        n_layers = automatically_determine_n_layers(event_shape)
    check_image_shape_for_multiscale_flow(event_shape, n_layers)

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
                 n_layers: int = None,
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
                 n_layers: int = None,
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
                 n_layers: int = None,
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
                 n_layers: int = None,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, LinearRational, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleDeepSigmoid(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = None,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, DeepSigmoid, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleDeepDenseSigmoid(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = None,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, DeepDenseSigmoid, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape


class MultiscaleDenseSigmoid(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = None,
                 factored: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, DenseSigmoid, n_layers, factored=factored)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape

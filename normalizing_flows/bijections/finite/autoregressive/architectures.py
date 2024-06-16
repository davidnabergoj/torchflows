from typing import Tuple, List, Type, Union

from normalizing_flows.bijections.finite.autoregressive.layers import (
    ShiftCoupling,
    AffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    RQSCoupling,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive,
    InverseAffineCoupling,
    DSCoupling,
    ElementwiseAffine,
    UMNNMaskedAutoregressive,
    LRSCoupling,
    LRSForwardMaskedAutoregressive
)
from normalizing_flows.bijections.base import BijectiveComposition
from normalizing_flows.bijections.finite.autoregressive.layers_base import CouplingBijection, \
    MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection
from normalizing_flows.bijections.finite.linear import ReversePermutation


def make_layers(base_bijection: Type[
    Union[CouplingBijection, MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection]],
                event_shape,
                n_layers: int = 2,
                edge_list: List[Tuple[int, int]] = None,
                image_coupling: bool = False):
    if image_coupling:
        if len(event_shape) == 2:
            bijections = make_image_layers_single_channel(base_bijection, event_shape, n_layers)
        elif len(event_shape) == 3:
            bijections = make_image_layers_multichannel(base_bijection, event_shape, n_layers)
        else:
            raise ValueError
    else:
        bijections = make_basic_layers(base_bijection, event_shape, n_layers, edge_list)
    return bijections


def make_basic_layers(base_bijection: Type[
    Union[CouplingBijection, MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection]],
                      event_shape,
                      n_layers: int = 2,
                      edge_list: List[Tuple[int, int]] = None):
    """
    Returns a list of bijections for transformations of vectors.
    """
    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers):
        if edge_list is None:
            bijections.append(ReversePermutation(event_shape=event_shape))
        bijections.append(base_bijection(event_shape=event_shape, edge_list=edge_list))
    bijections.append(ElementwiseAffine(event_shape=event_shape))
    return bijections


def make_image_layers_single_channel(base_bijection: Type[
    Union[CouplingBijection, MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection]],
                                     event_shape,
                                     n_layers: int = 2,
                                     checkerboard_resolution: int = 2):
    """
    Returns a list of bijections for transformations of images with a single channel.

    Each layer consists of two coupling transforms:
        1. checkerboard,
        2. checkerboard_inverted.
    """
    if len(event_shape) != 2:
        raise ValueError("Single-channel image transformation are only possible for inputs with two axes.")

    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers):
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'checkerboard',
                'resolution': checkerboard_resolution,
            }
        ))
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'checkerboard_inverted',
                'resolution': checkerboard_resolution,
            }
        ))
    bijections.append(ElementwiseAffine(event_shape=event_shape))
    return bijections


def make_image_layers_multichannel(base_bijection: Type[
    Union[CouplingBijection, MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection]],
                                   event_shape,
                                   n_layers: int = 2,
                                   checkerboard_resolution: int = 2):
    """
    Returns a list of bijections for transformations of images with multiple channels.

    Each layer consists of four coupling transforms:
        1. checkerboard,
        2. channel_wise,
        3. checkerboard_inverted,
        4. channel_wise_inverted.
    """
    if len(event_shape) != 3:
        raise ValueError("Multichannel image transformation are only possible for inputs with three axes.")

    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers):
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'checkerboard',
                'resolution': checkerboard_resolution,
            }
        ))
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'channel_wise'
            }
        ))
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'checkerboard_inverted',
                'resolution': checkerboard_resolution,
            }
        ))
        bijections.append(base_bijection(
            event_shape=event_shape,
            coupling_kwargs={
                'coupling_type': 'channel_wise_inverted'
            }
        ))
    bijections.append(ElementwiseAffine(event_shape=event_shape))
    return bijections


class NICE(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(ShiftCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class RealNVP(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(AffineCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class InverseRealNVP(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(InverseAffineCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class MAF(BijectiveComposition):
    """
    Expressive bijection with slightly unstable inverse due to autoregressive formulation.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(AffineForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class IAF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(AffineInverseMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingRQNSF(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(RQSCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveRQNSF(BijectiveComposition):
    """
    Expressive bijection with unstable inverse due to autoregressive formulation.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(RQSForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingLRS(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(LRSCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveLRS(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(LRSForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(RQSInverseMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingDSF(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 image_coupling: bool = False,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_layers(DSCoupling, event_shape, n_layers, edge_list, image_coupling)
        super().__init__(event_shape, bijections, **kwargs)


class UMNNMAF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 1, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(UMNNMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)

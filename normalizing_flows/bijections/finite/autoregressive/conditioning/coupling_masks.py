from typing import Tuple, List

import torch


class PartialCoupling:
    """
    Coupling mask object where a part of dimensions is kept unchanged and does not affect other dimensions.
    """

    def __init__(self,
                 event_shape,
                 source_mask: torch.Tensor,
                 target_mask: torch):
        """
        Partial coupling mask constructor.

        :param source_mask: boolean mask tensor of dimensions that affect target dimensions. This tensor has shape
         event_shape.
        :param target_mask: boolean mask tensor of affected dimensions. This tensor has shape event_shape.
        """
        self.event_shape = event_shape
        self.source_mask = source_mask
        self.target_mask = target_mask

        self.event_size = int(torch.prod(torch.as_tensor(self.event_shape)))

    @property
    def ignored_event_size(self):
        # Event size of ignored dimensions.
        return torch.sum(1 - (self.source_mask + self.target_mask))

    @property
    def source_event_size(self):
        return int(torch.sum(self.source_mask))

    @property
    def target_event_size(self):
        return int(torch.sum(self.target_mask))


class Coupling(PartialCoupling):
    """
    Base object which holds coupling partition mask information.
    """

    def __init__(self, event_shape, mask: torch.Tensor):
        super().__init__(event_shape, source_mask=mask, target_mask=~mask)

    @property
    def ignored_event_size(self):
        return 0


class GraphicalCoupling(PartialCoupling):
    def __init__(self, event_shape, edge_list: List[Tuple[int, int]]):
        if len(event_shape) != 1:
            raise ValueError("GraphicalCoupling is currently only implemented for vector data")

        source_dims = torch.tensor(sorted(list(set([e[0] for e in edge_list]))))
        target_dims = torch.tensor(sorted(list(set([e[1] for e in edge_list]))))

        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        source_mask = torch.isin(torch.arange(event_size), source_dims)
        target_mask = torch.isin(torch.arange(event_size), target_dims)

        super().__init__(event_shape, source_mask, target_mask)


class HalfSplit(Coupling):
    def __init__(self, event_shape):
        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        super().__init__(event_shape, mask=torch.less(torch.arange(event_size).view(*event_shape), event_size // 2))


class Checkerboard(Coupling):
    """
    Checkerboard coupling for image data.
    """

    def __init__(self, event_shape, resolution: int = 2, invert: bool = False):
        """
        :param event_shape: image shape with the form (n_channels, height, width). Note: width and height must be equal
        and a power of two.
        :param resolution: resolution of the checkerboard along one axis - the number of squares. Must be a power of two
        and smaller than image width.
        :param invert: invert the checkerboard mask.
        """
        height, width = event_shape[-2:]
        assert width % resolution == 0
        square_side_length = width // resolution
        assert resolution % 2 == 0
        half_resolution = resolution // 2
        a = torch.tensor([[1, 0] * half_resolution, [0, 1] * half_resolution] * half_resolution)
        mask = torch.kron(a, torch.ones((square_side_length, square_side_length)))
        mask = mask.bool()
        if invert:
            mask = ~mask
        super().__init__(event_shape, mask)


class ChannelWiseHalfSplit(Coupling):
    """
    Channel-wise coupling for image data.
    """

    def __init__(self, event_shape, invert: bool = False):
        """
        :param event_shape: image shape with the form (n_channels, height, width). Note: width and height must be equal
        and a power of two.
        :param invert: invert the checkerboard mask.
        """
        n_channels, height, width = event_shape
        mask = torch.as_tensor(torch.arange(start=0, end=n_channels) < (n_channels // 2))
        mask = mask[:, None, None].repeat(1, height, width)
        if invert:
            mask = ~mask
        super().__init__(event_shape, mask)


def make_coupling(event_shape, edge_list: List[Tuple[int, int]] = None, coupling_type: str = 'half_split', **kwargs):
    """

    :param event_shape:
    :param coupling_type: one of ['half_split', 'checkerboard', 'checkerboard_inverted', 'channel_wise',
        'channel_wise_inverted'].
    :param edge_list:
    :return:
    """
    if edge_list is not None:
        return GraphicalCoupling(event_shape, edge_list)
    else:
        if coupling_type == 'half_split':
            return HalfSplit(event_shape)
        elif coupling_type == 'checkerboard':
            return Checkerboard(event_shape, invert=False, **kwargs)
        elif coupling_type == 'checkerboard_inverted':
            return Checkerboard(event_shape, invert=True, **kwargs)
        elif coupling_type == 'channel_wise':
            return ChannelWiseHalfSplit(event_shape, invert=False)
        elif coupling_type == 'channel_wise_inverted':
            return ChannelWiseHalfSplit(event_shape, invert=True)
        else:
            raise ValueError

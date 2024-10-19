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
    def constant_shape(self) -> Tuple[int, ...]:
        return (self.source_event_size,)

    @property
    def target_event_size(self):
        return int(torch.sum(self.target_mask))

    @property
    def target_shape(self) -> Tuple[int, ...]:
        return (self.target_event_size,)


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
    elif coupling_type == 'half_split':
        return HalfSplit(event_shape)
    else:
        raise ValueError

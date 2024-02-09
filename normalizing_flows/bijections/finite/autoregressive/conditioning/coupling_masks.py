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


class HalfSplit(Coupling):
    def __init__(self, event_shape):
        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        super().__init__(event_shape, mask=torch.less(torch.arange(event_size).view(*event_shape), event_size // 2))


class GraphicalCoupling(PartialCoupling):
    def __init__(self, event_shape, edge_list: List[Tuple[int, int]]):
        source_mask = torch.tensor(sorted(list(set([e[0] for e in edge_list]))))
        target_mask = torch.tensor(sorted(list(set([e[1] for e in edge_list]))))
        super().__init__(event_shape, source_mask, target_mask)

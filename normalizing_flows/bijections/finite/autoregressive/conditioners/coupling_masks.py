import torch


class CouplingMask:
    """
    Base object which holds coupling partition mask information.
    """

    def __init__(self, event_shape):
        self.event_shape = event_shape
        self.event_size = int(torch.prod(torch.as_tensor(self.event_shape)))

    @property
    def mask(self):
        raise NotImplementedError

    @property
    def constant_event_size(self):
        raise NotImplementedError

    @property
    def transformed_event_size(self):
        raise NotImplementedError


class HalfSplit(CouplingMask):
    def __init__(self, event_shape):
        super().__init__(event_shape)
        self.event_partition_mask = torch.less(
            torch.arange(self.event_size).view(*self.event_shape),
            self.constant_event_size
        )

    @property
    def constant_event_size(self):
        return self.event_size // 2

    @property
    def transformed_event_size(self):
        return self.event_size - self.constant_event_size

    @property
    def mask(self):
        return self.event_partition_mask

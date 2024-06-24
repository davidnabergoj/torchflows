import torch

from normalizing_flows.bijections.finite.autoregressive.conditioning.coupling_masks import Coupling


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
        channels, height, width = event_shape
        assert width % resolution == 0
        square_side_length = width // resolution
        assert resolution % 2 == 0
        half_resolution = resolution // 2
        a = torch.tensor([[1, 0] * half_resolution, [0, 1] * half_resolution] * half_resolution)
        mask = torch.kron(a, torch.ones((square_side_length, square_side_length)))
        mask = mask.bool()
        mask = mask[None].repeat(channels, 1, 1)  # (channels, height, width)
        if invert:
            mask = ~mask
        self.resolution = resolution
        super().__init__(event_shape, mask)

    @property
    def constant_shape(self):
        n_channels, _, _ = self.event_shape
        return n_channels, self.resolution, self.resolution

    @property
    def transformed_shape(self):
        return self.constant_shape


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
        mask = mask[:, None, None].repeat(1, height, width)  # (channels, height, width)
        if invert:
            mask = ~mask
        super().__init__(event_shape, mask)

    @property
    def constant_shape(self):
        n_channels, height, width = self.event_shape
        return n_channels // 2, height, width

    @property
    def transformed_shape(self):
        n_channels, height, width = self.event_shape
        return n_channels - n_channels // 2, height, width


def make_image_coupling(event_shape, coupling_type: str, **kwargs):
    """

    :param event_shape:
    :param coupling_type: one of ['checkerboard', 'checkerboard_inverted', 'channel_wise', 'channel_wise_inverted'].
    :return:
    """
    if coupling_type == 'checkerboard':
        return Checkerboard(event_shape, invert=False, **kwargs)
    elif coupling_type == 'checkerboard_inverted':
        return Checkerboard(event_shape, invert=True, **kwargs)
    elif coupling_type == 'channel_wise':
        return ChannelWiseHalfSplit(event_shape, invert=False)
    elif coupling_type == 'channel_wise_inverted':
        return ChannelWiseHalfSplit(event_shape, invert=True)
    else:
        raise ValueError

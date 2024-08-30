import torch

from torchflows.bijections.finite.autoregressive.conditioning.coupling_masks import Coupling


class Checkerboard(Coupling):
    """
    Checkerboard coupling for image data.
    """

    def __init__(self, event_shape, invert: bool = False, **kwargs):
        """
        :param event_shape: image shape with the form (n_channels, height, width). Note: width and height must be equal
        and a power of two.
        :param invert: invert the checkerboard mask.
        """
        channels, height, width = event_shape
        mask = (torch.arange(height * width) % 2).view(height, width).bool()
        mask = mask[None].repeat(channels, 1, 1)  # (channels, height, width)
        if invert:
            mask = ~mask
        super().__init__(event_shape, mask)

    @property
    def constant_shape(self):
        n_channels, height, width = self.event_shape
        return n_channels, height // 2, width  # rectangular shape

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
        if n_channels <= 1:
            raise ValueError("Number of channels must be at least 2")

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

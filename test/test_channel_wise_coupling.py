import torch

from normalizing_flows.bijections.finite.multiscale.coupling import ChannelWiseHalfSplit


def test_partition_shapes_1():
    torch.manual_seed(0)
    image_shape = (3, 4, 4)
    coupling = ChannelWiseHalfSplit(image_shape, invert=True)
    assert coupling.constant_shape == (1, 4, 4)
    assert coupling.transformed_shape == (2, 4, 4)


def test_partition_shapes_2():
    torch.manual_seed(0)
    image_shape = (3, 16, 16)
    coupling = ChannelWiseHalfSplit(image_shape, invert=True)
    assert coupling.constant_shape == (1, 16, 16)
    assert coupling.transformed_shape == (2, 16, 16)

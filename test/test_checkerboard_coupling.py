import torch

from normalizing_flows.bijections.finite.multiscale.coupling import Checkerboard


def test_checkerboard_small():
    torch.manual_seed(0)
    image_shape = (3, 4, 4)
    coupling = Checkerboard(image_shape, resolution=2)
    assert torch.allclose(
        coupling.source_mask,
        torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.bool)[None].repeat(3, 1, 1)
    )
    assert torch.allclose(coupling.target_mask, ~coupling.source_mask)


def test_checkerboard_medium():
    torch.manual_seed(0)
    image_shape = (3, 16, 16)
    coupling = Checkerboard(image_shape, resolution=4)
    assert torch.allclose(
        coupling.source_mask,
        torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        ], dtype=torch.bool)[None].repeat(3, 1, 1)
    )
    assert torch.allclose(coupling.target_mask, ~coupling.source_mask)


def test_checkerboard_small_inverted():
    torch.manual_seed(0)
    image_shape = (3, 4, 4)
    coupling = Checkerboard(image_shape, resolution=2, invert=True)
    assert torch.allclose(
        coupling.source_mask,
        ~torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.bool)[None].repeat(3, 1, 1)
    )
    assert torch.allclose(coupling.target_mask, ~coupling.source_mask)


def test_partition_shapes_1():
    torch.manual_seed(0)
    image_shape = (3, 4, 4)
    coupling = Checkerboard(image_shape, resolution=2, invert=True)
    assert coupling.constant_shape == (3, 2, 2)
    assert coupling.transformed_shape == (3, 2, 2)


def test_partition_shapes_2():
    torch.manual_seed(0)
    image_shape = (3, 16, 16)
    coupling = Checkerboard(image_shape, resolution=8, invert=True)
    assert coupling.constant_shape == (3, 8, 8)
    assert coupling.transformed_shape == (3, 8, 8)


def test_partition_shapes_3():
    torch.manual_seed(0)
    image_shape = (3, 16, 8)
    coupling = Checkerboard(image_shape, resolution=4, invert=True)
    assert coupling.constant_shape == (3, 4, 4)
    assert coupling.transformed_shape == (3, 4, 4)

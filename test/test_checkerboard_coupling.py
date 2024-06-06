from normalizing_flows.bijections.finite.autoregressive.conditioning.coupling_masks import Checkerboard
import torch


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

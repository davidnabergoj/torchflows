import torch
from torchflows.bijections.continuous.rnode import RNODE


def test_difference():
    torch.manual_seed(0)
    batch_size = 3
    event_shape = (2,)

    x = torch.randn(size=(batch_size, *event_shape))

    b1 = RNODE(event_shape, atol=1e-1)
    b2 = RNODE(event_shape, atol=1e-2)

    z1, log_det_forward1 = b1.forward(x)
    z2, log_det_forward2 = b2.forward(x)

    assert not torch.allclose(z1, z2)
    assert not torch.allclose(log_det_forward1, log_det_forward2)

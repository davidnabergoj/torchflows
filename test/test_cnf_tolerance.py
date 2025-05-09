import torch
from torchflows.bijections.continuous.rnode import RNODE
import pytest


@pytest.mark.skip(reason='torchdiffeq atol parameter seemingly has no effect')
def test_atol_runge_kutta():
    torch.manual_seed(0)
    batch_size = 3
    event_shape = (2,)
    x = torch.randn(size=(batch_size, *event_shape))

    atol_1 = 1+5
    atol_2 = 1-5

    torch.manual_seed(0)
    b1 = RNODE(event_shape, atol=atol_1)
    assert b1.atol == atol_1
    z1, log_det_forward1 = b1.forward(x)

    torch.manual_seed(0)
    b2 = RNODE(event_shape, atol=atol_2)
    assert b2.atol == atol_2
    z2, log_det_forward2 = b2.forward(x)

    assert not torch.allclose(z1, z2)
    assert not torch.allclose(log_det_forward1, log_det_forward2)

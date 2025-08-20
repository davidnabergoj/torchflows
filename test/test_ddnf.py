from torchflows.bijections.continuous.ddnf import DeepDiffeomorphicBijection
import pytest
import torch


@pytest.mark.parametrize('n_steps', [1, 5, 10, 100, 200])
def test_n_steps(n_steps):
    torch.manual_seed(0)
    batch_size = 5
    event_shape = (10,)

    x = torch.randn(size=(batch_size, *event_shape))
    b = DeepDiffeomorphicBijection(event_shape=event_shape, n_steps=n_steps)
    z, log_det_forward = b.forward(x)
    
    assert z.shape == x.shape
    assert torch.isfinite(z).all()
    assert torch.isfinite(log_det_forward).all()

    xr, log_det_inverse = b.inverse(z)    
    assert xr.shape == x.shape
    assert torch.isfinite(xr).all()
    assert torch.isfinite(log_det_inverse).all()
from torchflows.bijections.continuous.base import HutchinsonTimeDerivative
from torchflows.bijections.continuous.ddnf import DDNF
from torchflows.bijections.continuous.ffjord import FFJORD
from torchflows.bijections.continuous.otflow import OTFlowBijection
from torchflows.bijections.continuous.rnode import RNODE
import torch
import pytest


@pytest.mark.parametrize('event_shape', [(1,), (2,), (5, 3, 2)])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_ot_flow(event_shape, batch_size):
    torch.manual_seed(0)
    x = torch.randn(size=(batch_size, *event_shape))
    b = OTFlowBijection(event_shape=event_shape)
    z, log_det_forward = b.forward(x)
    x2, log_det_inverse = b.inverse(z)


@pytest.mark.parametrize('event_shape', [(1,), (2,), (5, 3, 2)])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('cls', [RNODE, FFJORD, DDNF])
def test_approximate_trace_cnf(event_shape, batch_size, cls):
    torch.manual_seed(0)
    x = torch.randn(size=(batch_size, *event_shape))
    b = cls(event_shape=event_shape)
    z, log_det_forward = b.forward(x)
    x2, log_det_inverse = b.inverse(z)

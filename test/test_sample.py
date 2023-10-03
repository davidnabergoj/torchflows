import torch

from normalizing_flows import Flow
from normalizing_flows.bijections import RealNVP


def test_real_nvp():
    torch.manual_seed(0)
    f = Flow(RealNVP(event_shape=torch.Size((2, 3, 5, 7))))
    x = f.sample(10)
    assert x.shape == (10, 2, 3, 5, 7)

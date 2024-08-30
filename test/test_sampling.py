import pytest

from torchflows import Flow
from torchflows.architectures import PlanarFlow, SylvesterFlow, RadialFlow


@pytest.mark.parametrize('arch_cls', [PlanarFlow, SylvesterFlow, RadialFlow])
def test_basic(arch_cls):
    event_shape = (1, 2, 3, 4)
    f = Flow(arch_cls(event_shape=event_shape))
    assert f.sample((10,)).shape == (10, *event_shape)

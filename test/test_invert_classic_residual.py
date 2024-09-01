import pytest
import torch

from torchflows.flows import Flow
from torchflows.bijections.finite.residual.architectures import RadialFlow, SylvesterFlow, PlanarFlow


@pytest.mark.parametrize(
    'architecture_class',
    [
        RadialFlow,
        SylvesterFlow,
        PlanarFlow
    ]
)
def test_basic(architecture_class):
    torch.manual_seed(0)
    event_shape = (1, 2, 3, 4)
    batch_shape = (5, 6)

    flow = Flow(architecture_class(event_shape))
    x_new = flow.sample(batch_shape)
    assert x_new.shape == (*batch_shape, *event_shape)

    flow.bijection.invert()
    assert flow.log_prob(x_new).shape == batch_shape

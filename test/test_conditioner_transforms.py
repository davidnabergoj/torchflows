import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import (
    MADE, FeedForward, QuasiMADE, LinearMADE, ResidualFeedForward, Constant, Linear
)


@pytest.mark.parametrize('transform_class', [
    MADE,
    FeedForward,
    QuasiMADE,
    LinearMADE,
    ResidualFeedForward,
    Linear
])
def test_nontrivial_shape(transform_class):
    batch_shape = (3,)
    event_shape = (2, 4)
    n_parameters = 5

    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))

    transform = transform_class(input_event_shape=event_shape, output_event_shape=event_shape, n_predicted_parameters=n_parameters)
    out = transform(x)

    assert out.shape == (*batch_shape, *event_shape, n_parameters)


def test_constant_transform_nontrivial_shape():
    batch_shape = (3,)
    event_shape = (2, 4)
    n_parameters = 5

    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))

    transform = Constant(event_shape, n_parameters=n_parameters)
    out = transform(x)

    assert out.shape == (*batch_shape, *event_shape, n_parameters)

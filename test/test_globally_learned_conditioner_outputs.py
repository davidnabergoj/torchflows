import torch

from torchflows.bijections.finite.autoregressive.conditioning.transforms import FeedForward


def test_standard():
    torch.manual_seed(0)

    input_event_shape = torch.Size((10, 10))
    parameter_shape = torch.Size((20, 3))
    test_inputs = torch.randn(100, *input_event_shape)

    t = FeedForward(input_event_shape, parameter_shape)
    output = t(test_inputs)

    assert output.shape == (100, *parameter_shape)


def test_eighty_pct_global():
    torch.manual_seed(0)

    input_event_shape = torch.Size((10, 10))
    parameter_shape = torch.Size((20, 3))
    test_inputs = torch.randn(100, *input_event_shape)

    t = FeedForward(input_event_shape, parameter_shape, percentage_global_parameters=0.8)
    output = t(test_inputs)

    assert output.shape == (100, *parameter_shape)

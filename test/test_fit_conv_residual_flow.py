import pytest
import torch

from torchflows import Flow
from torchflows.architectures import ConvolutionalResFlow, ConvolutionalInvertibleResNet


@pytest.mark.parametrize('arch_cls', [ConvolutionalResFlow, ConvolutionalInvertibleResNet])
def test_basic(arch_cls):
    torch.manual_seed(0)
    event_shape = (3, 20, 20)
    flow = Flow(arch_cls(event_shape))
    flow.fit(torch.randn(size=(5, *event_shape)), n_epochs=20)

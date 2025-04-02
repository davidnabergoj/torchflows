import torch
from torch import nn

from torchflows import Flow
from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers import AffineCoupling, ElementwiseAffine
from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Affine
from torchflows.bijections.finite.multiscale.architectures import MultiscaleNICE
from torchflows.bijections.finite.multiscale.base import CheckerboardCoupling, NormalizedCheckerboardCoupling


def test_elementwise_affine():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(ElementwiseAffine(event_shape))
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_affine_coupling():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(AffineCoupling(event_shape))
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_checkerboard():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(CheckerboardCoupling(event_shape, Affine))
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_normalized_checkerboard():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(NormalizedCheckerboardCoupling(event_shape, transformer_class=Affine))
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_checkerboard_composition():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(BijectiveComposition([
        NormalizedCheckerboardCoupling(
            event_shape,
            transformer_class=Affine,
            alternate=i % 2 == 1,
            conditioner='convnet'
        )
        for i in range(4)
    ]))
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_multiscale_nice_small():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(MultiscaleNICE(event_shape, n_layers=1))
    assert isinstance(flow.bijection.checkerboard_layers, nn.ModuleList)
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None


def test_multiscale_nice():
    torch.manual_seed(0)
    event_shape = torch.Size((3, 20, 20))
    x = torch.randn(size=(4, *event_shape))
    flow = Flow(MultiscaleNICE(event_shape, n_layers=2))
    assert isinstance(flow.bijection.checkerboard_layers, nn.ModuleList)
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    loss.backward()
    assert loss.grad_fn is not None

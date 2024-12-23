from copy import deepcopy

import pytest
import torch

from torchflows import RNODE, Flow, Sylvester, RealNVP
from torchflows.bijections.base import invert


@pytest.mark.parametrize('flow_class', [RNODE, Sylvester, RealNVP])
def test_basic(flow_class):
    torch.manual_seed(0)
    b = flow_class(event_shape=(10,))
    deepcopy(b)


@pytest.mark.parametrize('flow_class', [RNODE, Sylvester, RealNVP])
def test_post_variational_fit(flow_class):
    torch.manual_seed(0)
    b = flow_class(event_shape=(10,))
    f = Flow(b)
    f.variational_fit(lambda x: torch.sum(-x ** 2), n_epochs=2)
    deepcopy(b)

@pytest.mark.parametrize('flow_class', [RNODE, Sylvester, RealNVP])
def test_post_fit(flow_class):
    torch.manual_seed(0)
    b = flow_class(event_shape=(10,))
    if isinstance(b, Sylvester):
        b = invert(b)
    f = Flow(b)
    f.fit(x_train=torch.randn(3, *b.event_shape), n_epochs=2)
    deepcopy(b)

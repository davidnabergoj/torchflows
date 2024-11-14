from copy import deepcopy

import torch

from torchflows import RNODE, Flow


def test_basic():
    torch.manual_seed(0)
    b = RNODE(event_shape=(10,))
    deepcopy(b)


def test_post_variational_fit():
    torch.manual_seed(0)
    b = RNODE(event_shape=(10,))
    f = Flow(b)
    f.variational_fit(lambda x: torch.sum(-x ** 2), n_epochs=2)
    b.eval()
    deepcopy(b)

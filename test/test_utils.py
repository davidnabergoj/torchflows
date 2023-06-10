from typing import Tuple

import pytest
import torch

from normalizing_flows.src.utils import get_batch_shape


@pytest.mark.parametrize('event_shape', [(1,), (2,), (3, 5), (5, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (3, 5), (5, 3), (1, 2, 3, 4)])
def test_batch_shape(event_shape: Tuple, batch_shape: Tuple):
    torch.manual_seed(0)
    event_shape = torch.Size(event_shape)
    batch_shape = torch.Size(batch_shape)
    x = torch.randn(*batch_shape, *event_shape)
    assert get_batch_shape(x, event_shape) == batch_shape

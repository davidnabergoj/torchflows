from typing import Union, Tuple

import torch

from normalizing_flows.src.bijections import Bijection


class Radial(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

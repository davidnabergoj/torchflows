from typing import Tuple

import torch

import torch.nn as nn


class Bijection(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Bijection class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward bijection map.
        Returns the output vector and the log jacobian determinant of the forward transform.

        :param x:
        :return:
        """
        raise NotImplementedError

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse bijection map.
        Returns the output vector and the log jacobian determinant of the inverse transform.

        :param z:
        :return:
        """
        raise NotImplementedError

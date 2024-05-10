import torch
import torch.distributions as dist
import torch.nn as nn


class TensorTrain(dist):
    """
    Class that defines a tensor train distribution.

    Reference paper: Khoo et al. "Tensorizing flows: a tool for variational inference" (2023), arxiv: 2305.02460.
    """

    def __init__(self, basis_size: int):
        assert basis_size > 0

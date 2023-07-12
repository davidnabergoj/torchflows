from typing import Tuple, Union

import torch


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]


def torch_func_nd(x: torch.Tensor, dim: Union[Tuple[int], int], func: callable, keeps_shape=True):
    """
    Apply torch function over several dimensions.
    This is achieved by placing all candidate dimensions at the end of the tensor via transposition,
    applying torch function, and transposing the tensor to its original shape.

    :param x: input tensor.
    :param dim: dimensions where function should be jointly applied.
    :return: tensor with function applied to specified dimensions.
    """
    # Permute x to place modified dimensions at the end of the tensor
    identity_dims = [i for i in range(len(x.shape)) if i not in dim]
    permuted_dims = identity_dims + list(dim)
    x = torch.permute(x, permuted_dims)

    permuted_shape = x.shape

    # Flatten the modified dimensions
    x = torch.flatten(x, start_dim=len(identity_dims), end_dim=len(x.shape) - 1)

    # Apply function on the flattened dimension
    x = func(x, dim=-1)

    if not keeps_shape:
        return x

    # Unflatten the last dimension
    x = x.view(*permuted_shape)

    # Permute x back into its original dimension order
    reverse_permuted_dims = torch.zeros(len(permuted_dims), dtype=torch.long)
    reverse_permuted_dims[permuted_dims] = torch.arange(len(x.shape), dtype=torch.long)
    x = torch.permute(x, list(reverse_permuted_dims))

    return x


def softmax_nd(x: torch.Tensor, dim: Union[Tuple[int], int]):
    """
    Apply softmax over several dimensions.
    This is achieved by placing all candidate dimensions at the end of the tensor via transposition,
    applying softmax, and transposing the tensor to its original shape.

    :param x: input tensor.
    :param dim: dimensions where softmax should be jointly applied.
    :return: tensor with softmax applied to specified dimensions.
    """
    if isinstance(dim, int):
        return torch.softmax(x, dim=dim)
    return torch_func_nd(x, dim, torch.softmax)


def log_softmax_nd(x: torch.Tensor, dim: Union[Tuple[int], int]):
    if isinstance(dim, int):
        return torch.log_softmax(x, dim=dim)
    return torch_func_nd(x, dim, torch.log_softmax)


def logsumexp_nd(x: torch.Tensor, dim: Union[Tuple[int], int]):
    if isinstance(dim, int):
        return torch.logsumexp(x, dim=dim)
    return torch_func_nd(x, dim, torch.logsumexp, keeps_shape=False)


def log_sigmoid(x: torch.Tensor):
    return -torch.nn.functional.softplus(-x)


def sum_except_batch(x, event_shape):
    return torch.sum(x, dim=list(range(len(x.shape)))[-len(event_shape):])


class GeometricBase(torch.distributions.Geometric):
    def __init__(self, *args, **kwargs):
        """
        Support: [0, inf).
        """
        super().__init__(*args, **kwargs)

    def cdf(self, value):
        return torch.clip(1 - (1 - self.probs) ** (torch.floor(value.float()) + 1), 0.0)


class Geometric(GeometricBase):
    def __init__(self, minimum: int = 1, *args, **kwargs):
        # Support: [minimum, inf)
        super().__init__(*args, **kwargs)
        self.minimum = minimum

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return super().cdf(value - self.minimum)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape=sample_shape) + self.minimum

    def log_prob(self, value):
        return super().log_prob(value - self.minimum)

from typing import Tuple, Union

import torch


def vjp_tensor(v: torch.Tensor, y: torch.Tensor, x: torch.Tensor, **kwargs):
    """
    Compute the vector-Jacobian product v.T @ J where v is an input tensor and J is the Jacobian of y = f(x) at x.
    Tensors v and y have the same shape.
    Tensor x must have requires_grad set to True.

    :param v: input with shape "event_shape".
    :param y: function output to be differentiated.
    :param x: function input with shape "event_shape".
    :return: output tensor v.T @ J.
    """
    assert x.requires_grad
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=v,
        is_grads_batched=False,
        **kwargs
    )[0]


def flatten_event(x: torch.Tensor, event_shape: torch.Size):
    """
    Flattens event dimensions of x into a single dimension.

    :param x: input tensor with shape (*batch_shape, *event_shape).
    :param event_shape: input tensor event shape.
    :return: output tensor with shape (*batch_shape, n_event_dims) where n_event_dims = prod(event_shape).
    """
    batch_shape = get_batch_shape(x, event_shape)
    return x.view(*batch_shape, -1)


def unflatten_event(x: torch.Tensor, event_shape: torch.Size):
    """
    Flattens last dimension of x into a specified event shape.

    :param x: input tensor with shape (*batch_shape, n_event_dims).
    :param event_shape: output tensor event shape.
    :return: output tensor with shape (*batch_shape, *event_shape) where n_event_dims = prod(event_shape).
    """
    batch_shape = x.shape[:-1]
    return x.view(*batch_shape, *event_shape)


def flatten_batch(x: torch.Tensor, batch_shape: torch.Size):
    """
    Flattens batch dimensions of x into a single dimension.

    :param x: input tensor with shape (*batch_shape, *event_shape).
    :param batch_shape: input tensor batch shape.
    :return: output tensor with shape (n_batch_dims, *event_shape) where n_batch_dims = prod(batch_shape).
    """
    n_batch_dims = len(batch_shape)
    event_shape = x.shape[n_batch_dims:]
    return x.view(-1, *event_shape)


def unflatten_batch(x: torch.Tensor, batch_shape: torch.Size):
    """
    Flattens first dimension of x into a specified batch shape.

    :param x: input tensor with shape (n_batch_dims, *event_shape).
    :param batch_shape: output tensor batch shape.
    :return: output tensor with shape (*batch_shape, *event_shape) where n_batch_dims = prod(batch_shape).
    """
    event_shape = x.shape[1:]
    return x.view(*batch_shape, *event_shape)


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

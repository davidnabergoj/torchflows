import torch
import torch.nn as nn

from torchflows.utils import flatten_event


class ContextCombiner(nn.Module):
    """
    Abstract helper class that combines an input tensor with a context tensor.
    """

    def __init__(self, input_shape, context_shape):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape
        self.n_input_dims = int(
            torch.prod(torch.as_tensor(self.input_shape))
        ) if input_shape is not None else None
        self.n_context_dims = int(
            torch.prod(torch.as_tensor(self.context_shape))
        ) if context_shape is not None else None

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        x.shape = (*batch_shape, *event_shape)
        context.shape = (*batch_shape, *context_shape)
        output.shape = (*batch_shape, self.n_output_dims)

        self.n_output_dims depends on the context combination strategy. For concatenation: n_event_dims + n_context_dims
        """
        raise NotImplementedError

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError


class Concatenation(ContextCombiner):
    """
    Concatenates context to the input.
    """

    def __init__(self, input_shape, context_shape):
        super().__init__(input_shape=input_shape, context_shape=context_shape)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        x_flat = flatten_event(x, self.input_shape)
        context_flat = flatten_event(context, self.context_shape)
        return torch.cat([x_flat, context_flat], dim=-1)

    @property
    def n_output_dims(self) -> int:
        return self.n_input_dims + self.n_context_dims


class Bypass(ContextCombiner):
    """
    Ignores the context.
    """

    def __init__(self, input_shape):
        super().__init__(
            input_shape=input_shape,
            context_shape=None
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        return flatten_event(x, self.input_shape)

    @property
    def n_output_dims(self) -> int:
        return self.n_input_dims

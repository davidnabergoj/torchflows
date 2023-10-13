import math

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.autoregressive.conditioners.context import Concatenation, ContextCombiner, \
    Bypass
from normalizing_flows.utils import get_batch_shape, pad_leading_dims


class ConditionerTransform(nn.Module):
    def __init__(self,
                 input_event_shape,
                 context_shape,
                 output_event_shape,
                 n_predicted_parameters: int,
                 context_combiner: ContextCombiner = None):
        super().__init__()
        if context_shape is None:
            context_combiner = Bypass(input_event_shape)
        elif context_shape is not None and context_combiner is None:
            context_combiner = Concatenation(input_event_shape, context_shape)
        self.context_combiner = context_combiner

        # The conditioner transform receives as input the context combiner output
        self.input_event_shape = input_event_shape
        self.output_event_shape = output_event_shape
        self.context_shape = context_shape
        self.n_input_event_dims = self.context_combiner.n_output_dims
        self.n_output_event_dims = int(torch.prod(torch.as_tensor(output_event_shape)))
        self.n_predicted_parameters = n_predicted_parameters

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x.shape = (*batch_shape, *input_event_shape)
        # context.shape = (*batch_shape, *context_shape)
        # output.shape = (*batch_shape, *output_event_shape, n_predicted_parameters)
        raise NotImplementedError


class Constant(ConditionerTransform):
    def __init__(self, output_event_shape, n_parameters: int, fill_value: float = None):
        super().__init__(
            input_event_shape=None,
            context_shape=None,
            output_event_shape=output_event_shape,
            n_predicted_parameters=n_parameters
        )
        if fill_value is None:
            initial_theta = torch.randn(size=(*self.output_event_shape, n_parameters,))
        else:
            initial_theta = torch.full(size=(*self.output_event_shape, n_parameters), fill_value=fill_value)
        self.theta = nn.Parameter(initial_theta)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        n_batch_dims = len(x.shape) - len(self.output_event_shape)
        n_event_dims = len(self.output_event_shape)
        batch_shape = x.shape[:n_batch_dims]
        return pad_leading_dims(self.theta, n_batch_dims).repeat(*batch_shape, *([1] * n_event_dims), 1)


class MADE(ConditionerTransform):
    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self,
                 input_event_shape: torch.Size,
                 output_event_shape: torch.Size,
                 n_predicted_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            output_event_shape=output_event_shape,
            n_predicted_parameters=n_predicted_parameters
        )

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        # Set conditional dimension values
        ms = [
            torch.arange(self.n_input_event_dims) + 1,
            *[(torch.arange(n_hidden) % (self.n_input_event_dims - 1)) + 1 for _ in range(n_layers - 1)],
            torch.arange(self.n_output_event_dims) + 1
        ]

        # Create autoencoder masks
        masks = self.create_masks(n_layers, ms)

        layers = []
        for mask in masks[:-1]:
            n_layer_outputs, n_layer_inputs = mask.shape
            layers.extend([self.MaskedLinear(n_layer_inputs, n_layer_outputs, mask), nn.Tanh()])

        # Final linear layer
        layers.extend([
            self.MaskedLinear(
                masks[-1].shape[1],
                masks[-1].shape[0] * n_predicted_parameters,
                torch.repeat_interleave(masks[-1], n_predicted_parameters, dim=0)
            ),
            nn.Unflatten(dim=-1, unflattened_size=(*output_event_shape, n_predicted_parameters))
        ])
        self.sequential = nn.Sequential(*layers)

    @staticmethod
    def create_masks(n_layers, ms):
        masks = []
        for i in range(1, n_layers + 1):
            m_current = ms[i]
            m_previous = ms[i - 1]
            xx, yy = torch.meshgrid(m_current, m_previous, indexing='ij')
            if i == n_layers:
                masks.append(torch.as_tensor(xx > yy, dtype=torch.float))
            else:
                masks.append(torch.as_tensor(xx >= yy, dtype=torch.float))
        return masks

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        out = self.sequential(self.context_combiner(x, context))
        batch_shape = get_batch_shape(x, self.input_event_shape)
        return out.view(*batch_shape, *self.output_event_shape, self.n_predicted_parameters)


class LinearMADE(MADE):
    def __init__(self, input_event_shape: torch.Size, output_event_shape: torch.Size, n_predicted_parameters: int,
                 **kwargs):
        super().__init__(input_event_shape, output_event_shape, n_predicted_parameters, n_layers=1, **kwargs)


class FeedForward(ConditionerTransform):
    def __init__(self,
                 input_event_shape: torch.Size,
                 output_event_shape: torch.Size,
                 n_predicted_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            output_event_shape=output_event_shape,
            n_predicted_parameters=n_predicted_parameters
        )

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_event_dims += self.n_context_dims

        layers = []

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * n_predicted_parameters))
        elif n_layers > 1:
            layers.extend([nn.Linear(self.n_input_event_dims, n_hidden), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.Tanh()])
            layers.append(nn.Linear(n_hidden, self.n_output_event_dims * n_predicted_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(*output_event_shape, n_predicted_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        out = self.sequential(self.context_combiner(x, context))
        batch_shape = get_batch_shape(x, self.input_event_shape)
        return out.view(*batch_shape, *self.output_event_shape, self.n_predicted_parameters)


class Linear(FeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, n_layers=1)


class ResidualFeedForward(ConditionerTransform):
    class ResidualLinear(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.linear = nn.Linear(n_in, n_out)

        def forward(self, x):
            return x + self.linear(x)

    def __init__(self,
                 input_event_shape: torch.Size,
                 output_event_shape: torch.Size,
                 n_predicted_parameters: int,
                 context_shape: torch.Size = None,
                 n_layers: int = 2):
        super().__init__(input_event_shape, context_shape, output_event_shape, n_predicted_parameters)

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_event_dims += self.n_context_dims

        layers = []

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * n_predicted_parameters))
        elif n_layers > 1:
            layers.extend([self.ResidualLinear(self.n_input_event_dims, self.n_input_event_dims), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([self.ResidualLinear(self.n_input_event_dims, self.n_input_event_dims), nn.Tanh()])
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * n_predicted_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(*output_event_shape, n_predicted_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        out = self.sequential(self.context_combiner(x, context))
        batch_shape = get_batch_shape(x, self.input_event_shape)
        return out.view(*batch_shape, *self.output_event_shape, self.n_predicted_parameters)

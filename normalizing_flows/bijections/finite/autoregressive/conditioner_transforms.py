import math

import torch
import torch.nn as nn


class ConditionerTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x.shape = (*batch_shape, *event_shape)
        # context.shape = (*batch_shape, *event_shape)
        # output.shape = (*batch_shape, *event_shape, n_parameters)
        raise NotImplementedError


class Constant(ConditionerTransform):
    def __init__(self, event_shape, n_parameters: int, fill_value: float = None):
        super().__init__()
        self.event_shape = event_shape
        if fill_value is None:
            initial_theta = torch.randn(size=(*self.event_shape, n_parameters,))
        else:
            initial_theta = torch.full(size=(*self.event_shape, n_parameters), fill_value=fill_value)
        self.theta = nn.Parameter(initial_theta)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        n_batch_dims = len(x.shape) - len(self.event_shape)
        n_event_dims = len(self.event_shape)
        batch_shape = x.shape[:n_batch_dims]
        return self.theta[[None] * n_batch_dims].repeat(*batch_shape, *([1] * n_event_dims), 1)


class MADE(ConditionerTransform):
    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self,
                 input_shape: torch.Size,
                 output_shape: torch.Size,
                 n_output_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2):
        super().__init__()
        self.context_shape = context_shape

        self.n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        self.n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        self.n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        if n_hidden is None:
            n_hidden = int(3 * math.log10(self.n_input_dims))

        # Set conditional dimension values
        ms = [
            torch.arange(self.n_input_dims) + 1,
            *[(torch.arange(n_hidden) % (self.n_input_dims - 1)) + 1 for _ in range(n_layers - 1)],
            torch.arange(self.n_output_dims) + 1
        ]

        # Create autoencoder masks
        masks = self.create_masks(n_layers, ms)

        layers = [nn.Flatten(start_dim=-len(input_shape))]  # First layer flattens the input
        for mask in masks[:-1]:
            n_layer_outputs, n_layer_inputs = mask.shape
            layers.extend([self.MaskedLinear(n_layer_inputs, n_layer_outputs, mask), nn.Tanh()])

        # Final linear layer
        layers.extend([
            self.MaskedLinear(
                masks[-1].shape[1],
                masks[-1].shape[0] * n_output_parameters,
                torch.repeat_interleave(masks[-1], n_output_parameters, dim=0)
            ),
            nn.Unflatten(dim=-1, unflattened_size=(*output_shape, n_output_parameters))
        ])
        self.sequential = nn.Sequential(*layers)

        if context_shape is not None:
            self.context_linear = nn.Sequential(
                nn.Linear(self.n_context_dims, self.n_output_dims * n_output_parameters),
                nn.Unflatten(dim=-1, unflattened_size=(self.n_output_dims, n_output_parameters))
            )

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
        out = self.sequential(x)
        if context is not None:
            assert x.shape[0] == context.shape[0], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            out += self.context_linear(context)
        if context is None and self.context_shape is not None:
            raise RuntimeError("Context required")
        return out


class LinearMADE(MADE):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, n_output_parameters: int, **kwargs):
        super().__init__(input_shape, output_shape, n_output_parameters, n_layers=1, **kwargs)


class QuasiMADE(MADE):
    # https://arxiv.org/pdf/2009.07419.pdf
    @staticmethod
    def create_masks(n_layers, ms):
        masks = []
        for i in range(1, n_layers + 1):
            m_current = ms[i]
            m_previous = ms[i - 1]
            xx, yy = torch.meshgrid(m_current, m_previous, indexing='ij')
            masks.append(torch.as_tensor(xx >= yy, dtype=torch.float))
        return masks

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # TODO modify autograd
        raise NotImplementedError


class FeedForward(ConditionerTransform):
    def __init__(self,
                 input_shape: torch.Size,
                 output_shape: torch.Size,
                 n_output_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape

        self.n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        self.n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        self.n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        if n_hidden is None:
            n_hidden = int(3 * math.log10(self.n_input_dims))

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_dims += self.n_context_dims

        layers = [nn.Flatten(start_dim=-len(input_shape))]  # First layer flattens the input

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_dims, self.n_output_dims * n_output_parameters))
        elif n_layers > 1:
            layers.extend([nn.Linear(self.n_input_dims, n_hidden), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.Tanh()])
            layers.append(nn.Linear(n_hidden, self.n_output_dims * n_output_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(*output_shape, n_output_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        if context is not None:
            assert x.shape[:len(self.input_shape)] == context.shape[:len(self.input_shape)], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            x = torch.cat([x, context], dim=-len(self.input_shape))
        if context is None and self.context_shape is not None:
            raise RuntimeError("Context required")
        return self.sequential(x)


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
                 input_shape: torch.Size,
                 output_shape: torch.Size,
                 n_output_parameters: int,
                 context_shape: torch.Size = None,
                 n_layers: int = 2):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape

        self.n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        self.n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        self.n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_dims += self.n_context_dims

        layers = [nn.Flatten(start_dim=-len(input_shape))]  # First layer flattens the input

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_dims, self.n_output_dims * n_output_parameters))
        elif n_layers > 1:
            layers.extend([self.ResidualLinear(self.n_input_dims, self.n_input_dims), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([self.ResidualLinear(self.n_input_dims, self.n_input_dims), nn.Tanh()])
            layers.append(nn.Linear(self.n_input_dims, self.n_output_dims * n_output_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(*output_shape, n_output_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        if context is not None:
            assert x.shape[:len(self.input_shape)] == context.shape[:len(self.input_shape)], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            x = torch.cat([x, context], dim=-len(self.input_shape))
        if context is None and self.context_shape is not None:
            raise RuntimeError("Context required")
        return self.sequential(x)
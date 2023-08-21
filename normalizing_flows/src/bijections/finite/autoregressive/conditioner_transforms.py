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
    def __init__(self, n_output_parameters: int):
        super().__init__()
        self.parameters = nn.Parameter(torch.zeros(size=(n_output_parameters,)))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.parameters[[None] * len(x.shape)].repeat(*x.shape, 1)


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
                 n_hidden: int = 100,
                 n_layers: int = 2):
        super().__init__()
        self.context_shape = context_shape

        n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        # Set conditional dimension values
        ms = [
            torch.arange(n_input_dims) + 1,
            *[(torch.arange(n_hidden) % (n_input_dims - 1)) + 1 for _ in range(n_layers - 1)],
            torch.arange(n_output_dims) + 1
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
                masks[-1].shape[0] * n_output_parameters,
                torch.repeat_interleave(masks[-1], n_output_parameters, dim=0)
            ),
            nn.Unflatten(dim=-1, unflattened_size=(n_output_dims, n_output_parameters))
        ])
        self.sequential = nn.Sequential(*layers)

        if context_shape is not None:
            self.context_linear = nn.Sequential(
                nn.Linear(n_context_dims, n_output_dims * n_output_parameters),
                nn.Unflatten(dim=-1, unflattened_size=(n_output_dims, n_output_parameters))
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
        pass


class FeedForward(ConditionerTransform):
    def __init__(self,
                 input_shape: torch.Size,
                 output_shape: torch.Size,
                 n_output_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = 100,
                 n_layers: int = 2):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape

        n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        # If context given, concatenate it to transform input
        if context_shape is not None:
            n_input_dims += n_context_dims

        # Check the one layer special case
        if n_layers == 1:
            layers = [nn.Linear(n_input_dims, n_output_dims * n_output_parameters)]
        elif n_layers > 1:
            layers = [nn.Linear(n_input_dims, n_hidden), nn.Tanh()]
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.Tanh()])
            layers.append(nn.Linear(n_hidden, n_output_dims * n_output_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(n_output_dims, n_output_parameters)))
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

        n_input_dims = int(torch.prod(torch.as_tensor(input_shape)))
        n_output_dims = int(torch.prod(torch.as_tensor(output_shape)))
        n_context_dims = int(torch.prod(torch.as_tensor(context_shape))) if context_shape is not None else None

        # If context given, concatenate it to transform input
        if context_shape is not None:
            n_input_dims += n_context_dims

        # Check the one layer special case
        if n_layers == 1:
            layers = [nn.Linear(n_input_dims, n_output_dims * n_output_parameters)]
        elif n_layers > 1:
            layers = [self.ResidualLinear(n_input_dims, n_input_dims), nn.Tanh()]
            for _ in range(n_layers - 2):
                layers.extend([self.ResidualLinear(n_input_dims, n_input_dims), nn.Tanh()])
            layers.append(nn.Linear(n_input_dims, n_output_dims * n_output_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(n_output_dims, n_output_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        if context is not None:
            assert x.shape[:len(self.input_shape)] == context.shape[:len(self.input_shape)], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            x = torch.cat([x, context], dim=-len(self.input_shape))
        if context is None and self.context_shape is not None:
            raise RuntimeError("Context required")
        return self.sequential(x)

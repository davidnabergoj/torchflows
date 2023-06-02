import torch
import torch.nn as nn


class ConditionerTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        raise NotImplementedError


class MADE(ConditionerTransform):
    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self,
                 n_input_dims: int,
                 n_output_dims: int,
                 n_output_parameters: int,
                 n_context_dims: int = None,
                 n_hidden: int = 100,
                 n_layers: int = 4):
        super().__init__()

        # Set conditional dimension values
        ms = [
            torch.arange(n_input_dims) + 1,
            *[(torch.arange(n_hidden) % (n_input_dims - 1)) + 1 for _ in range(n_layers - 1)],
            torch.arange(n_output_dims) + 1
        ]

        # Create autoencoder masks
        masks = []
        for i in range(1, n_layers + 1):
            m_current = ms[i]
            m_previous = ms[i - 1]
            xx, yy = torch.meshgrid(m_current, m_previous, indexing='ij')
            if i == n_layers:
                masks.append(torch.as_tensor(xx > yy, dtype=torch.float))
            else:
                masks.append(torch.as_tensor(xx >= yy, dtype=torch.float))

        layers = []
        for mask in masks[:-1]:
            n_layer_outputs, n_layer_inputs = mask.shape
            layers.extend([self.MaskedLinear(n_layer_inputs, n_layer_outputs, mask), nn.ReLU()])

        # Final linear layer
        layers.extend([
            self.MaskedLinear(
                masks[-1].shape[1],
                masks[-1].shape[0] * n_output_parameters,
                torch.repeat_interleave(masks[-1], n_output_parameters, dim=0)
            ),
            nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters))
        ])
        self.sequential = nn.Sequential(*layers)

        if n_context_dims is not None:
            self.context_linear = nn.Sequential(
                nn.Linear(n_context_dims, n_output_dims * n_output_parameters),
                nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters))
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        out = self.sequential(x)
        if context is not None:
            assert x.shape[0] == context.shape[0], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            out += self.context_linear(context)
        return out


class LinearMADE(MADE):
    def __init__(self, n_input_dims: int, n_output_dims: int, n_output_parameters: int, **kwargs):
        super().__init__(n_input_dims, n_output_dims, n_output_parameters, n_layers=1, **kwargs)


class FeedForward(ConditionerTransform):
    def __init__(self,
                 n_input_dims: int,
                 n_output_dims: int,
                 n_output_parameters: int,
                 n_context_dims: int = None,
                 n_hidden: int = 100,
                 n_layers: int = 4):
        super().__init__()

        # If context given, concatenate it to transform input
        if n_context_dims is not None:
            n_input_dims += n_context_dims

        # Check the one layer special case
        if n_layers == 1:
            layers = [nn.Linear(n_input_dims, n_output_dims * n_output_parameters)]
        elif n_layers > 1:
            layers = [nn.Linear(n_input_dims, n_hidden), nn.ReLU()]
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
            layers.append(nn.Linear(n_hidden, n_output_dims * n_output_parameters))
        else:
            raise ValueError

        # Reshape the output
        layers.append(nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters)))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        if context is not None:
            assert x.shape[0] == context.shape[0], \
                f"Batch shapes of x and context must match ({x.shape = }, {context.shape = })"
            x = torch.cat([x, context], dim=1)
        return self.sequential(x)


class Linear(FeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, n_layers=1)

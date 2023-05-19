import torch
import torch.nn as nn


class MADE(nn.Sequential):
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
                 n_hidden: int = 100,
                 n_layers: int = 4):

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
            xx, yy = torch.meshgrid(m_current, m_previous)
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
        super().__init__(*layers)


class LinearMADE(MADE):
    def __init__(self, n_input_dims: int, n_output_dims: int, n_output_parameters: int, **kwargs):
        super().__init__(n_input_dims, n_output_dims, n_output_parameters, n_layers=1, **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self,
                 n_input_dims: int,
                 n_output_dims: int,
                 n_output_parameters: int,
                 n_hidden: int = 100,
                 n_layers: int = 4):
        layers = [nn.Linear(n_input_dims, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        layers.append(nn.Linear(n_hidden, n_output_dims * n_output_parameters))
        layers.append(nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters)))
        super().__init__(*layers)


class Linear(nn.Sequential):
    def __init__(self,
                 n_input_dims: int,
                 n_output_dims: int,
                 n_output_parameters: int):
        super().__init__(
            nn.Linear(n_input_dims, n_output_dims * n_output_parameters),
            nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters))
        )

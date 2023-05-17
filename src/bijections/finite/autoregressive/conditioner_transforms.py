import torch
import torch.nn as nn


class DeepMADE(nn.Sequential):
    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    @staticmethod
    def create_masks(n_inputs: int,
                     n_outputs: int,
                     n_dim: int,
                     n_hidden: int,
                     n_layers: int):
        ms = [
            torch.randint(low=1, high=n_dim, size=(n_inputs,)),
            *[torch.randint(low=1, high=n_dim, size=(n_hidden,)) for _ in range(n_layers - 1)],
            torch.randint(low=1, high=n_dim, size=(n_outputs,)),
        ]

        masks = []
        for i in range(1, n_layers + 1):
            m_current = ms[i]
            m_previous = ms[i - 1]
            xx, yy = torch.meshgrid(m_current, m_previous)
            masks.append(torch.as_tensor(xx >= yy, dtype=torch.float))

        return masks, ms[0]

    def __init__(self,
                 n_input_dims: int,
                 n_output_dims: int,
                 n_output_parameters: int,
                 n_hidden: int = 100,
                 n_layers: int = 4):
        masks, self.input_degrees = self.create_masks(
            n_inputs=n_input_dims,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_outputs=n_output_dims,
            n_dim=n_input_dims
        )
        layers = [DeepMADE.MaskedLinear(n_input_dims, n_hidden, masks[0]), nn.ReLU()]
        for i in range(1, n_layers - 1):
            layers.extend([DeepMADE.MaskedLinear(n_hidden, n_hidden, masks[i]), nn.ReLU()])
        layers.append(DeepMADE.MaskedLinear(
            n_hidden,
            n_output_dims * n_output_parameters,
            masks[-1].repeat(n_output_parameters, 1))
        )
        layers.append(nn.Unflatten(dim=1, unflattened_size=(n_output_dims, n_output_parameters)))
        super().__init__(*layers)


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

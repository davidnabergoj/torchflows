import torch
import torch.nn as nn


class MADE(nn.Module):
    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


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

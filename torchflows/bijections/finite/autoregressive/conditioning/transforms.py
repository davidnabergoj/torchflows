import math
from typing import Tuple, Union, Type

import torch
import torch.nn as nn

from torchflows.bijections.finite.autoregressive.conditioning.context import Concatenation, ContextCombiner, \
    Bypass
from torchflows.utils import get_batch_shape


class ConditionerTransform(nn.Module):
    """
    Module which predicts transformer parameters for the transformation of a tensor y using an input tensor x and
     possibly a corresponding context tensor c.

    In other words, a conditioner transform f predicts theta = f(x, c) to be used in transformer g with z = g(y; theta).
    The transformation g is performed elementwise on tensor y.
    Since g transforms each element of y with a parameter tensor of shape (n_transformer_parameters,),
     the shape of theta is (*y.shape, n_transformer_parameters).
    """

    def __init__(self,
                 input_event_shape,
                 context_shape,
                 parameter_shape: Union[torch.Size, Tuple[int, ...]],
                 context_combiner: ContextCombiner = None,
                 global_parameter_mask: torch.Tensor = None,
                 initial_global_parameter_value: float = None,
                 **kwargs):
        """
        :param input_event_shape: shape of conditioner input tensor x.
        :param context_shape: shape of conditioner context tensor c.
        :param parameter_shape: shape of parameter tensor required to transform transformer input y.
        :param context_combiner: ContextCombiner class which defines how to combine x and c to predict theta.
        :param global_parameter_mask: boolean tensor which determines which elements of parameter tensors should be
        learned globally instead of predicted. If an element is set to 1, that element is learned globally.
         We require that global_parameter_mask.shape = parameter_shape.
        :param initial_global_parameter_value: initial global parameter value as a single scalar. If None, all initial
         global parameters are independently drawn from the standard normal distribution.
        """
        super().__init__()
        if global_parameter_mask is not None and global_parameter_mask.shape != parameter_shape:
            raise ValueError(
                f"Global parameter mask must have shape equal to the output parameter shape {parameter_shape}, "
                f"but found {global_parameter_mask.shape}"
            )

        if context_shape is None:
            context_combiner = Bypass(input_event_shape)
        elif context_shape is not None and context_combiner is None:
            context_combiner = Concatenation(input_event_shape, context_shape)
        self.context_combiner = context_combiner

        # The conditioner transform receives as input the context combiner output
        self.input_event_shape = input_event_shape
        self.context_shape = context_shape
        self.n_input_event_dims = self.context_combiner.n_output_dims

        # Setup output parameter attributes
        self.parameter_shape = parameter_shape
        self.global_parameter_mask = global_parameter_mask
        self.n_transformer_parameters = int(torch.prod(torch.as_tensor(self.parameter_shape)))
        self.n_global_parameters = 0 if global_parameter_mask is None else int(torch.sum(self.global_parameter_mask))
        self.n_predicted_parameters = self.n_transformer_parameters - self.n_global_parameters

        if initial_global_parameter_value is None:
            initial_global_theta_flat = torch.randn(size=(self.n_global_parameters,))
        else:
            initial_global_theta_flat = torch.full(
                size=(self.n_global_parameters,),
                fill_value=initial_global_parameter_value
            )
        self.global_theta_flat = nn.Parameter(initial_global_theta_flat)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x.shape = (*batch_shape, *self.input_event_shape)
        # context.shape = (*batch_shape, *self.context_shape)
        # output.shape = (*batch_shape, *self.parameter_shape)
        batch_shape = get_batch_shape(x, self.input_event_shape)
        if self.n_global_parameters == 0:
            # All parameters are predicted
            return self.predict_theta_flat(x, context).view(*batch_shape, *self.parameter_shape)
        else:
            if self.n_global_parameters == self.n_transformer_parameters:
                # All transformer parameters are learned globally
                output = torch.zeros(*batch_shape, *self.parameter_shape, device=x.device)
                output[..., self.global_parameter_mask] = self.global_theta_flat
                return output
            else:
                # Some transformer parameters are learned globally, some are predicted
                output = torch.zeros(*batch_shape, *self.parameter_shape, device=x.device)
                output[..., self.global_parameter_mask] = self.global_theta_flat
                output[..., ~self.global_parameter_mask] = self.predict_theta_flat(x, context)
                return output

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None):
        raise NotImplementedError

    def regularization(self):
        return sum([torch.sum(torch.square(p)) for p in self.parameters()])


class Constant(ConditionerTransform):
    def __init__(self, event_shape, parameter_shape, fill_value: float = None):
        super().__init__(
            input_event_shape=event_shape,
            context_shape=None,
            parameter_shape=parameter_shape,
            initial_global_parameter_value=fill_value,
            global_parameter_mask=torch.ones(parameter_shape, dtype=torch.bool)
        )


class MADE(ConditionerTransform):
    """
    Masked autoencoder for distribution estimation (MADE).

    MADE is a conditioner transform that receives as input a tensor x. It predicts parameters for the
     transformer such that each dimension only depends on the previous ones.
    """

    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self,
                 input_event_shape: Union[torch.Size, Tuple[int, ...]],
                 output_event_shape: Union[torch.Size, Tuple[int, ...]],
                 parameter_shape_per_element: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 n_hidden: int = None,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            parameter_shape=(*output_event_shape, *parameter_shape_per_element),
            **kwargs
        )
        n_predicted_parameters_per_element = int(torch.prod(torch.as_tensor(parameter_shape_per_element)))
        n_output_event_dims = int(torch.prod(torch.as_tensor(output_event_shape)))

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        # Set conditional dimension values
        ms = [
            torch.arange(self.n_input_event_dims) + 1,
            *[(torch.arange(n_hidden) % (self.n_input_event_dims - 1)) + 1 for _ in range(n_layers - 1)],
            torch.arange(n_output_event_dims) + 1
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
                masks[-1].shape[0] * n_predicted_parameters_per_element,
                torch.repeat_interleave(masks[-1], n_predicted_parameters_per_element, dim=0)
            )
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

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None):
        theta = self.sequential(self.context_combiner(x, context))
        # (*b, *e, *pe)

        if self.global_parameter_mask is None:
            return torch.flatten(theta, start_dim=len(theta.shape) - len(self.input_event_shape))
        else:
            return theta[..., ~self.global_parameter_mask]


class LinearMADE(MADE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_layers=1, **kwargs)


class FeedForward(ConditionerTransform):
    def __init__(self,
                 input_event_shape: torch.Size,
                 parameter_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2,
                 nonlinearity: Type[nn.Module] = nn.Tanh,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            parameter_shape=parameter_shape,
            **kwargs
        )

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_event_dims, self.n_predicted_parameters))
        elif n_layers > 1:
            layers.extend([nn.Linear(self.n_input_event_dims, n_hidden), nonlinearity()])
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nonlinearity()])
            layers.append(nn.Linear(n_hidden, self.n_predicted_parameters))
        else:
            raise ValueError
        layers.append(nn.Unflatten(dim=-1, unflattened_size=self.parameter_shape))
        self.sequential = nn.Sequential(*layers)

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.sequential(self.context_combiner(x, context))


class Linear(FeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, n_layers=1)


class ResidualFeedForward(ConditionerTransform):
    class ResidualBlock(nn.Module):
        def __init__(self, event_size: int, hidden_size: int, block_size: int, nonlinearity: Type[nn.Module]):
            super().__init__()
            if block_size < 2:
                raise ValueError(f"block_size must be at least 2 but found {block_size}. "
                                 f"For block_size = 1, use the FeedForward class instead.")
            layers = []
            layers.extend([nn.Linear(event_size, hidden_size), nonlinearity()])
            for _ in range(block_size - 2):
                layers.extend([nn.Linear(hidden_size, hidden_size), nonlinearity()])
            layers.extend([nn.Linear(hidden_size, event_size)])
            self.sequential = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.sequential(x)

    def __init__(self,
                 input_event_shape: torch.Size,
                 parameter_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 3,
                 block_size: int = 2,
                 nonlinearity: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            parameter_shape=parameter_shape,
            **kwargs
        )

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        if n_layers <= 2:
            raise ValueError(f"Number of layers in ResidualFeedForward must be at least 3, but found {n_layers}")

        layers = [nn.Linear(self.n_input_event_dims, n_hidden), nonlinearity()]
        for _ in range(n_layers - 2):
            layers.append(self.ResidualBlock(n_hidden, n_hidden, block_size, nonlinearity=nonlinearity))
        layers.append(nn.Linear(n_hidden, self.n_predicted_parameters))
        layers.append(nn.Unflatten(dim=-1, unflattened_size=self.parameter_shape))
        self.sequential = nn.Sequential(*layers)

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.sequential(self.context_combiner(x, context))


class CombinedConditioner(nn.Module):
    """
    Class that uses two different conditioners (each acting on different dimensions) to predict transformation
    parameters. Transformation parameters are combined in a single vector.
    """

    def __init__(self,
                 conditioner1: ConditionerTransform,
                 conditioner2: ConditionerTransform,
                 conditioner1_input_mask: torch.Tensor,
                 conditioner2_input_mask: torch.Tensor):
        super().__init__()
        self.conditioner1 = conditioner1
        self.conditioner2 = conditioner2
        self.mask1 = conditioner1_input_mask
        self.mask2 = conditioner2_input_mask

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        h1 = self.conditioner1(x[..., self.mask1], context)
        h2 = self.conditioner1(x[..., self.mask1], context)
        return h1 + h2

    def regularization(self):
        return self.conditioner1.regularization() + self.conditioner2.regularization()


class RegularizedCombinedConditioner(CombinedConditioner):
    def __init__(self,
                 conditioner1: ConditionerTransform,
                 conditioner2: ConditionerTransform,
                 conditioner1_input_mask: torch.Tensor,
                 conditioner2_input_mask: torch.Tensor,
                 regularization_coefficient_1: float,
                 regularization_coefficient_2: float):
        super().__init__(
            conditioner1,
            conditioner2,
            conditioner1_input_mask,
            conditioner2_input_mask
        )
        self.c1 = regularization_coefficient_1
        self.c2 = regularization_coefficient_2

    def regularization(self):
        return self.c1 * self.conditioner1.regularization() + self.c2 * self.conditioner2.regularization()


class RegularizedGraphicalConditioner(RegularizedCombinedConditioner):
    def __init__(self,
                 interacting_dimensions_conditioner: ConditionerTransform,
                 auxiliary_dimensions_conditioner: ConditionerTransform,
                 interacting_dimensions_mask: torch.Tensor,
                 auxiliary_dimensions_mask: torch.Tensor,
                 coefficient: float = 0.1):
        super().__init__(
            interacting_dimensions_conditioner,
            auxiliary_dimensions_conditioner,
            interacting_dimensions_mask,
            auxiliary_dimensions_mask,
            regularization_coefficient_1=0.0,
            regularization_coefficient_2=coefficient
        )

import math

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.autoregressive.conditioners.context import Concatenation, ContextCombiner, \
    Bypass
from normalizing_flows.utils import get_batch_shape, pad_leading_dims


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
                 n_transformer_parameters: int,
                 context_combiner: ContextCombiner = None,
                 percent_global_parameters: float = 0.0,
                 initial_global_parameter_value: float = None):
        """
        :param input_event_shape: shape of conditioner input tensor x.
        :param context_shape: shape of conditioner context tensor c.
        :param n_transformer_parameters: number of parameters required to transform a single element of y.
        :param context_combiner: ContextCombiner class which defines how to combine x and c to predict theta.
        :param percent_global_parameters: percent of all parameters in theta to be learned independent of x and c.
         A value of 0 means the conditioner predicts n_transformer_parameters parameters based on x and c.
         A value of 1 means the conditioner predicts no parameters based on x and c, but outputs globally learned theta.
         A value of alpha means the conditioner outputs alpha * n_transformer_parameters parameters globally and
         predicts the rest. In this case, the predicted parameters are the last alpha * n_transformer_parameters
         elements in theta.
        :param initial_global_parameter_value: the initial value for the entire globally learned part of theta. If None,
         the global part of theta is initialized to samples from the standard normal distribution.
        """
        super().__init__()
        if context_shape is None:
            context_combiner = Bypass(input_event_shape)
        elif context_shape is not None and context_combiner is None:
            context_combiner = Concatenation(input_event_shape, context_shape)
        self.context_combiner = context_combiner

        # The conditioner transform receives as input the context combiner output
        self.input_event_shape = input_event_shape
        self.context_shape = context_shape
        self.n_input_event_dims = self.context_combiner.n_output_dims
        self.n_transformer_parameters = n_transformer_parameters
        self.n_global_parameters = int(n_transformer_parameters * percent_global_parameters)
        self.n_predicted_parameters = self.n_transformer_parameters - self.n_global_parameters

        if initial_global_parameter_value is None:
            initial_global_theta = torch.randn(size=(self.n_global_parameters,))
        else:
            initial_global_theta = torch.full(
                size=(self.n_global_parameters,),
                fill_value=initial_global_parameter_value
            )
        self.global_theta = nn.Parameter(initial_global_theta)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Compute parameters theta for each input tensor.
        This includes globally learned parameters and parameters which are predicted based on x and context.

        :param x: batch of input tensors with x.shape = (*batch_shape, *self.input_event_shape).
        :param context: batch of context tensors with context.shape = (*batch_shape, *self.context_shape).
        :return: batch of parameter tensors theta with theta.shape = (*batch_shape, self.n_transformer_parameters).
        """
        if self.n_global_parameters == 0:
            return self.predict_theta(x, context)
        else:
            batch_shape = get_batch_shape(x, self.input_event_shape)
            n_batch_dims = len(batch_shape)
            batch_global_theta = pad_leading_dims(self.global_theta, n_batch_dims).repeat(*batch_shape, 1)
            if self.n_global_parameters == self.n_transformer_parameters:
                return batch_global_theta
            else:
                return torch.cat([batch_global_theta, self.predict_theta(x, context)], dim=-1)

    def predict_theta(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Predict parameters theta for each input tensor.
        Note: this method does not set any global parameters, but instead only predicts parameters from x and context.

        :param x: batch of input tensors with x.shape = (*batch_shape, *self.input_event_shape).
        :param context: batch of context tensors with context.shape = (*batch_shape, *self.context_shape).
        :return: batch of parameter tensors theta with theta.shape = (*batch_shape, self.n_predicted_parameters).
        """
        raise NotImplementedError


class Constant(ConditionerTransform):
    """
    Constant conditioner transform, which only uses global parameters theta and no local parameters.
    """

    def __init__(self, input_event_shape, n_parameters: int, fill_value: float = None):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=None,
            n_transformer_parameters=n_parameters,
            initial_global_parameter_value=fill_value,
            percent_global_parameters=1.0
        )


class MADE(ConditionerTransform):
    """
    Masked autoencoder for distribution estimation.
    """

    class MaskedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
            super().__init__(in_features=in_features, out_features=out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def __init__(self,
                 input_event_shape: torch.Size,

                 n_transformer_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            n_transformer_parameters=n_transformer_parameters,
            **kwargs
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
                masks[-1].shape[0] * self.n_predicted_parameters,
                torch.repeat_interleave(masks[-1], self.n_predicted_parameters, dim=0)
            ),
            nn.Unflatten(dim=-1, unflattened_size=(self.n_predicted_parameters,))
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

    def predict_theta(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.sequential(self.context_combiner(x, context))


class LinearMADE(MADE):
    """
    Masked autoencoder for distribution estimation with a single layer.
    """

    def __init__(self, input_event_shape: torch.Size, n_transformer_parameters: int,
                 **kwargs):
        super().__init__(
            input_event_shape,
            n_transformer_parameters,
            n_layers=1,
            **kwargs
        )


class FeedForward(ConditionerTransform):
    """
    Feed-forward neural network conditioner transform.
    """

    def __init__(self,
                 input_event_shape: torch.Size,
                 n_transformer_parameters: int,
                 context_shape: torch.Size = None,
                 n_hidden: int = None,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=context_shape,
            n_transformer_parameters=n_transformer_parameters,
            **kwargs
        )

        if n_hidden is None:
            n_hidden = max(int(3 * math.log10(self.n_input_event_dims)), 4)

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_event_dims += self.n_context_dims

        layers = []

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * n_transformer_parameters))
        elif n_layers > 1:
            layers.extend([nn.Linear(self.n_input_event_dims, n_hidden), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.Tanh()])
            layers.append(nn.Linear(n_hidden, self.n_output_event_dims * self.n_predicted_parameters))
        else:
            raise ValueError

        self.sequential = nn.Sequential(*layers)

    def predict_theta(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.sequential(self.context_combiner(x, context))


class Linear(FeedForward):
    """
    Linear conditioner transform with the map: theta = a * combiner(x, context) + b.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, n_layers=1)


class ResidualFeedForward(ConditionerTransform):
    """
    Residual feed-forward neural network conditioner transform.
    """

    class ResidualLinear(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.linear = nn.Linear(n_in, n_out)

        def forward(self, x):
            return x + self.linear(x)

    def __init__(self,
                 input_event_shape: torch.Size,
                 n_transformer_parameters: int,
                 context_shape: torch.Size = None,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__(
            input_event_shape,
            context_shape,
            n_transformer_parameters,
            **kwargs
        )

        # If context given, concatenate it to transform input
        if context_shape is not None:
            self.n_input_event_dims += self.n_context_dims

        layers = []

        # Check the one layer special case
        if n_layers == 1:
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * self.n_predicted_parameters))
        elif n_layers > 1:
            layers.extend([self.ResidualLinear(self.n_input_event_dims, self.n_input_event_dims), nn.Tanh()])
            for _ in range(n_layers - 2):
                layers.extend([self.ResidualLinear(self.n_input_event_dims, self.n_input_event_dims), nn.Tanh()])
            layers.append(nn.Linear(self.n_input_event_dims, self.n_output_event_dims * self.n_predicted_parameters))
        else:
            raise ValueError

        self.sequential = nn.Sequential(*layers)

    def predict_theta(self, x: torch.Tensor, context: torch.Tensor = None):
        return self.sequential(self.context_combiner(x, context))

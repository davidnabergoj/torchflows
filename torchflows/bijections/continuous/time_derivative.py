# This file incorporates work covered by the following copyright and permission notice:
#
#   MIT License
#
#   Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.


# This file is an adaptation of code from the following repository https://github.com/rtqichen/ffjord

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class TimeDerivative(nn.Module):
    """Base class to compute time derivatives.

    Subclasses only need to implement the step method.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # number of function evaluations
        self.register_buffer('_n_evals', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def prepare_initial_state(self,
                              z0: torch.Tensor,
                              div0: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Prepare the initial state tuple, containing all relevant tensors.
        For example, the state delta and the divergence (integrated into state and log_det).
        Result may contain additional tensors to be integrated.

        :param torch.Tensor z0: initial state delta with shape `(batch_size, event_size)`.
        :param torch.Tensor div0: initial divergence with shape `(batch_size, 1)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor, ...].
        :return: initial state tuple. All tensors have initial dimension equal to `batch_size`.
        """
        raise NotImplementedError

    def forward(self,
                t: torch.Tensor,
                state: Tuple[torch.Tensor, ...]):
        """Compute next state.

        :param torch.Tensor t: delta time tensor with shape `()`.
        :param Tuple[torch.Tensor, ...] state: state tuple with tensors: the event (space) tensor, the divergence 
            tensor, and possible auxiliary tensors. Each tensor has the first dimension equal to `(batch_size,)`.
        :rtype: Tuple[torch.Tensor, ...].
        :return: new state with the same structure as `state`.
        """
        self._n_evals += 1
        with torch.enable_grad():
            state[0].requires_grad_(True)
            t.requires_grad_(True)
            for a in state[2:]:
                a.requires_grad_(True)
            dxdt, div, *aux = self.step(t, state[0], *state[2:])
            new_state = (dxdt, div, *aux)
            if len(state) != len(new_state):
                raise ValueError(
                    f"Current state contains {len(state)} elements, but new state contains {len(new_state)}"
                )
            return new_state

    def step(self,
             t: torch.Tensor,
             x: torch.Tensor,
             *aux: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Perform one ODE step: compute dx/dt and the corresponding divergence.

        TODO double check input and output shapes (event_size vs *event_shape).

        :param torch.Tensor t: time tensor with shape `()`.
        :param torch.Tensor x: spatial tensor with shape `(batch_size, event_size)`.
        :param Tuple[torch.Tensor, ...] aux: auxiliary data such as hutchinson noise samples for divergence estimation.
        :rtype: Tuple[torch.Tensor, torch.Tensor, ...].
        :return: dx/dt tensor with shape `(batch_size, event_size)`, divergence tensor with shape `(batch_size,)`, and 
         possible auxiliary tensors with shape `(batch_size, ...)`.
        """
        raise NotImplementedError

    def regularization(self):
        return torch.tensor(0.0)

    def before_odeint(self, **kwargs):
        self._n_evals.fill_(0)

class TimeDerivativeModule(nn.Module):
    """Base class for time derivative modules.
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Compute the derivative of x wrt t.

        :param torch.Tensor t: time tensor with shape `()`.
        :param torch.Tensor x: space tensor with shape `(batch_size, *event_shape)`.
        :rtype: torch.Tensor.
        :return: time derivative tensor with shape `(batch_size, *event_shape)`
        """
        raise NotImplementedError

    def sq_norm_param(self) -> torch.Tensor:
        """Return the squared norm of trainable parameters.

        :rtype: torch.Tensor.
        :return: squared norm of parameters as a tensor with shape `()`.
        """
        return sum([
            torch.sum(torch.square(p))
            for p in self.parameters()
            if p.requires_grad
        ])

    def regularization(self):
        """Compute regularization.

        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`. 
        """
        return torch.tensor(0.0)

class TimeDerivativeSequential(TimeDerivativeModule):
    def __init__(self, layers: List[TimeDerivativeModule]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, 
                t: torch.Tensor, 
                x: torch.Tensor):
        for layer in self.layers:
            x = layer.forward(t, x)
        return x

class HyperLinear(TimeDerivativeModule):
    """
    Apply y(t) = A(t) @ x + b(t)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 hypernet_dim: int = 8,
                 n_hidden: int = 1,
                 activation: nn.Module = nn.Tanh):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(TimeDerivativeModule):
    """
    Apply y = A @ x + b without any time information.
    """

    def __init__(self,
                 input_shape: Union[Tuple[int, ...], torch.Size],
                 output_shape: Union[Tuple[int, ...], torch.Size]):
        super().__init__()
        self.input_shape = input_shape
        self.input_size = int(torch.prod(torch.as_tensor(input_shape)))
        self.output_shape = output_shape
        self.output_size = int(torch.prod(torch.as_tensor(output_shape)))
        self._layer = nn.Linear(self.input_size, self.output_size)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        x_flat = x.view(-1, self.input_size)
        out_flat = self._layer(x_flat)
        return out_flat.view(-1, *self.output_shape)


class ConcatLinear(TimeDerivativeModule):
    """
    Apply y = A @ [t; x] + b
    """

    def __init__(self,
                 input_shape: Union[Tuple[int, ...], torch.Size],
                 output_shape: Union[Tuple[int, ...], torch.Size]):
        super().__init__()
        self.input_shape = input_shape
        self.input_size = int(torch.prod(torch.as_tensor(input_shape)))
        self.output_shape = output_shape
        self.output_size = int(torch.prod(torch.as_tensor(output_shape)))
        self._layer = nn.Linear(self.input_size + 1, self.output_size)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        # x.shape = (b, *input_shape)
        x_flat = x.view(-1, self.input_size)
        tt_flat = torch.ones_like(x_flat[:, :1]) * t
        ttx_flat = torch.cat([tt_flat, x_flat], 1)
        out_flat = self._layer(ttx_flat)
        return out_flat.view(-1, *self.output_shape)


class ConcatLinear_v2(TimeDerivativeModule):
    """
    Apply y = (A @ x + b) + (D @ t + e)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(TimeDerivativeModule):
    """
    Apply y = (A @ x + b) * sigmoid(D @ t + e)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))


class ConcatSquashLinear(TimeDerivativeModule):
    """
    Apply y = (A @ x + b) * sigmoid(D @ t + e) + (F @ t)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class HyperConv2d(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(x, W(t), b(t))
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out *
                          self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(
                self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(
                self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )


class IgnoreConv2d(TimeDerivativeModule):
    """
    Apply y = Conv2d(x, W, b)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x)


class SquashConv2d(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(x, W, b) * sigmoid(A @ t + c)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(concatenate([x, t]), W, b)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(x, W, b) + (A @ t + c)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(x, W, b) * sigmoid(A @ t + c) + (D @ t)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(1, -1, 1, 1) \
            + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(TimeDerivativeModule):
    """
    Apply y(t) = Conv2d(concatenate([x, t, coords_height, coords_width]), W, b)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: bool = 3,
                 stride: bool = 1,
                 padding: bool = 0,
                 dilation: bool = 1,
                 groups: bool = 1,
                 bias: bool = True,
                 transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(TimeDerivativeModule):
    """
    Apply y = (A @ x + b) * sigmoid(D @ x + e)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(TimeDerivativeModule):
    """
    Apply y = Conv2d(W @ x + b) * sigmoid(Conv2d(A @ x + c))
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1):
        super().__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(TimeDerivativeModule):
    """
    Apply y = Conv2dTranspose(W @ x + b) * sigmoid(Conv2dTranspose(A @ x + c))
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 output_padding: int = 0,
                 groups: int = 1):
        super().__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(TimeDerivativeModule):
    """
    Apply y = (A @ x + b) + (D @ x + e) * t
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 layer_class: nn.Module = nn.Linear,
                 **kwargs):
        """
        :param kwargs: unused.
        """
        super().__init__()
        self._layer0 = layer_class(dim_in, dim_out)
        self._layer1 = layer_class(dim_in, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(TimeDerivativeModule):
    """
    Apply y = Conv2d(x, A, b) + Conv2d(x, D, e) * t
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 ksize: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transpose: bool = False,
                 **kwargs):
        """
        :param kwargs: unused.
        """
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t

from typing import Tuple, List, Union, Any

import torch

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchflows.utils import get_batch_shape


class Bijection(nn.Module):
    """Bijection class.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 **kwargs):
        """Bijection constructor.

        :param event_shape: shape of the event tensor.
        :param context_shape: shape of the context tensor.
        :param kwargs: unused.
        """
        super().__init__()
        self.event_shape = event_shape
        self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        self.context_shape = context_shape

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward bijection map.
        Returns the output vector and the log Jacobian determinant of the forward transform.

        :param torch.Tensor x: input array with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor context: context array with shape `(*batch_shape, *context_shape)`.
        :return: output array and log determinant. The output array has shape `(*batch_shape, *event_shape)`; the log
            determinant has shape `(*batch_shape,)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse bijection map.
        Returns the output vector and the log Jacobian determinant of the inverse transform.

        :param z: input array with shape `(*batch_shape, *event_shape)`.
        :param context: context array with shape `(*batch_shape, *context_shape)`.
        :return: output array and log determinant. The output array has shape `(*batch_shape, *event_shape)`; the log
            determinant has shape `(*batch_shape,)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    def batch_apply(self, fn, batch_size, x, context=None):
        batch_shape = x.shape[:-len(self.event_shape)]

        if context is None:
            x_flat = torch.flatten(x, start_dim=0, end_dim=len(batch_shape) - 1)
            dataset = TensorDataset(x_flat)
        else:
            x_flat = torch.flatten(x, start_dim=0, end_dim=len(batch_shape) - 1)
            context_flat = torch.flatten(context, start_dim=0, end_dim=len(batch_shape) - 1)
            dataset = TensorDataset(x_flat, context_flat)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        log_dets = []
        for batch in data_loader:
            batch_out, batch_log_det = fn(*batch)
            outputs.append(batch_out)
            log_dets.append(batch_log_det)
        outputs = torch.cat(outputs, dim=0).view_as(x)
        log_dets = torch.cat(log_dets, dim=0).view(*batch_shape)
        return outputs, log_dets

    def batch_forward(self, x: torch.Tensor, batch_size: int, context: torch.Tensor = None):
        # TODO remove the if statement, context=None is the default anyway
        if context:
            outputs, log_dets = self.batch_apply(self.forward, batch_size, x, context)
        else:
            outputs, log_dets = self.batch_apply(self.forward, batch_size, x)
        assert outputs.shape == x.shape
        batch_shape = get_batch_shape(x, self.event_shape)
        assert log_dets.shape == batch_shape
        return outputs, log_dets

    def batch_inverse(self, x: torch.Tensor, batch_size: int, context: torch.Tensor = None):
        # TODO remove the if statement, context=None is the default anyway
        if context:
            outputs, log_dets = self.batch_apply(self.inverse, batch_size, x, context)
        else:
            outputs, log_dets = self.batch_apply(self.inverse, batch_size, x)
        assert outputs.shape == x.shape
        batch_shape = get_batch_shape(x, self.event_shape)
        assert log_dets.shape == batch_shape
        return outputs, log_dets

    def regularization(self):
        return 0.0

    def invert(self):
        self.forward, self.inverse = self.inverse, self.forward


def invert(bijection: Bijection) -> Bijection:
    """Swap the forward and inverse methods of the input bijection.

    :param Bijection bijection: bijection to be inverted.
    :returns: inverted bijection.
    :rtype: Bijection
    """
    bijection.forward, bijection.inverse = bijection.inverse, bijection.forward
    return bijection


class BijectiveComposition(Bijection):
    """
    Composition of bijections. Inherits from Bijection.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 layers: List[Bijection],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 **kwargs):
        """
        BijectiveComposition constructor.

        :param event_shape: shape of the event tensor.
        :param List[Bijection] layers: bijection layers.
        :param context_shape: shape of the context tensor.
        :param kwargs: unused.
        """
        super().__init__(event_shape=event_shape, context_shape=context_shape)
        self.layers = nn.ModuleList(layers)

    def freeze_after(self, index: int):
        """
        Freeze all layers after the given index (exclusive).

        :param int index: index after which the layers are frozen.
        """
        for i in range(len(self.layers)):
            if i > index:
                self.layers[i].requires_grad_(False)

    def unfreeze_all_layers(self):
        for i in range(len(self.layers)):
            self.layers[i].requires_grad_(True)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(size=get_batch_shape(x, event_shape=self.event_shape)).to(x)
        for layer in self.layers:
            try:
                x, log_det_layer = layer(x, context=context)
            except TypeError as e:
                print(e)
                print(layer)
                raise e
            log_det += log_det_layer
        z = x
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(size=get_batch_shape(z, event_shape=self.event_shape)).to(z)
        for layer in self.layers[::-1]:
            z, log_det_layer = layer.inverse(z, context=context)
            log_det += log_det_layer
        x = z
        return x, log_det

    def regularization(self):
        return sum([layer.regularization() for layer in self.layers])

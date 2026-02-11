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

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """Forward bijection map.
        Returns the output vector and the log Jacobian determinant of the forward transform.

        :param torch.Tensor x: input array with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor context: context array with shape `(*batch_shape, *context_shape)`.
        :rtype: Tuple[torch.Tensor, ...].
        :return: output array and log determinant. The output array has shape `(*batch_shape, *event_shape)`; the log
            determinant has shape `batch_shape`.
        """
        raise NotImplementedError

    def inverse(self, 
                z: torch.Tensor, 
                context: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """Inverse bijection map.
        Returns the output vector and the log Jacobian determinant of the inverse transform.

        :param torch.Tensor z: input array with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor context: context array with shape `(*batch_shape, *context_shape)`.
        :rtype: Tuple[torch.Tensor, ...].
        :return: output array and log determinant. The output array has shape `(*batch_shape, *event_shape)`; the log
            determinant has shape `batch_shape`.
        """
        raise NotImplementedError

    def batch_apply(self, 
                    fn: callable, 
                    batch_size: int, 
                    x: torch.Tensor, 
                    context: torch.Tensor = None,
                    **kwargs) -> Tuple[torch.Tensor, ...]:
        """Apply a function to an input tensor in batches.
        The function can return a tuple of tensors.

        :param callable fn: function to be applied. Receives as input a tensor with shape `(*batch_shape, *event_shape)`
         and outputs a tuple of tensors, each with batch dimensions equal to `batch_shape`.
        :param int batch_size: size of batches sent to `fn`.
        :param torch.Tensor x: data tensor with shape `(*batch_shape, *event_shape)`.
        :param Optional[torch.Tensor] context: optional context tensor with shape `(*batch_shape, *context_shape)`.
        :param kwargs: keyword arguments sent to `fn`.
        :rtype: Tuple[torch.Tensor, ...].
        :return: tuple of tensors with batch dimensions equal to `batch_shape`.
        """
        batch_shape = x.shape[:-len(self.event_shape)]

        if context is None:
            x_flat = torch.flatten(x, start_dim=0, end_dim=len(batch_shape) - 1)
            dataset = TensorDataset(x_flat)
        else:
            x_flat = torch.flatten(x, start_dim=0, end_dim=len(batch_shape) - 1)
            context_flat = torch.flatten(context, start_dim=0, end_dim=len(batch_shape) - 1)
            dataset = TensorDataset(x_flat, context_flat)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        output_lists = []
        for batch in data_loader:
            batch_out = fn(*batch, **kwargs)
            for i, element in enumerate(batch_out):
                if len(output_lists) < (i + 1):
                    output_lists.append([])
                output_lists[i].append(element)
        return tuple([torch.cat(lst, dim=0) for lst in output_lists])

    def batch_forward(self,
                      x: torch.Tensor,
                      batch_size: int,
                      context: torch.Tensor = None,
                      **kwargs):
        return self.batch_apply(
            self.forward, 
            batch_size, 
            x, 
            context,
            **kwargs
        )

    def batch_inverse(self, 
                      x: torch.Tensor,
                      batch_size: int,
                      context: torch.Tensor = None,
                      **kwargs):
        return self.batch_apply(
            self.inverse, 
            batch_size, 
            x, 
            context,
            **kwargs
        )

    def regularization(self, *aux: Tuple[Any, ...]) -> torch.Tensor:
        """Compute regularization.

        The default regularization is 0, but can be overriden in subclasses.

        :param Tuple[Any, ...] aux: auxiliary data.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`. 
        """
        return torch.tensor(0.0)

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

    def regularization(self, *aux):
        """Compute regularization.

        :param Tuple[Any, ...] aux: unused.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`. 
        """
        return torch.tensor(0.0)

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

    def __init__(self, layers: List[Bijection], **kwargs):
        """
        BijectiveComposition constructor.

        :param event_shape: shape of the event tensor.
        :param List[Bijection] layers: bijection layers.
        :param kwargs: unused.
        """
        super().__init__(
            event_shape=layers[0].event_shape,
            context_shape=layers[0].context_shape
        )
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

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param kwargs: keyword arguments passed to the forward method of each layer.
        """
        log_det = torch.zeros(size=get_batch_shape(x, event_shape=self.event_shape)).to(x)
        for layer in self.layers:
            try:
                x, log_det_layer = layer.forward(
                    x, 
                    context=context,
                    **kwargs
                )
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
        """Compute regularization for a composition of bijections.

        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`.
        """
        total = torch.tensor(0.0)
        for i in range(len(self.layers)):
            total += self.layers[i].regularization()
        return total

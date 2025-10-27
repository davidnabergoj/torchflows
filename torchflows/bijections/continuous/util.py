from typing import Union, Tuple, List
import torch


def delta_sq_norm_jac(e_dzdx: torch.Tensor):
    """

    :param torch.Tensor e_dzdx: tensor with shape `(batch_size, n_samples, event_size)`.
    :rtype: torch.Tensor.
    :return: delta of the squared Jacobian norm with shape `(batch_size,)`.
    """
    # sqnorms = []
    # for e_dzdx in e_dzdx_vals:
    #     n = e_dzdx.pow(2).mean(dim=1, keepdim=True)
    #     sqnorms.append(n)
    # return torch.cat(sqnorms, dim=1).mean(dim=1)
    # e_dzdx = torch.stack(e_dzdx_vals, dim=1)  # (batch, num_events, event_size)
    return e_dzdx.pow(2).mean(dim=(1, 2))


def approximate_divergence(dz: torch.Tensor,
                           x: torch.Tensor,
                           e: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
                           return_e_dzdx: bool = False):
    """Compute divergence term for ODE integration (integrates to log determinant).
    Approximation divergence as the Hessian trace.

    TODO check shapes (event_size vs *event_shape)

    :param torch.Tensor dz: time derivative tensor (dx/dt) with shape `(batch_size, event_size)`.
    :param torch.Tensor dx: delta of the state, tensor with shape `(batch_size, event_size)`.
    :param torch.Tensor e: tuple of one or more Hutchinson noise samples, each sample has shape `(batch_size, event_size)`.
    :param bool return_e_dzdx: if True, return intermediate e_dzdx terms, useful for e.g., the delta of the squared 
     Jacobian norm computation.
    :rtype: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor].
    :return: divergence term tensor. If return_e_dzdx is True, also return a stacked list of those terms.
    """
    if isinstance(e, torch.Tensor):
        e = (e,)  # Handle e being a single sample
    elif not isinstance(e, tuple):
        raise ValueError(
            "Hutchinson noise must be passed in as a tuple or torch.Tensor."
        )

    samples = []
    e_dzdx_vals = []
    for e_ in e:
        e_dzdx = torch.autograd.grad(dz, x, e_, create_graph=True)[0]
        if return_e_dzdx:
            e_dzdx_vals.append(e_dzdx)
        e_dzdx_e = e_dzdx * e_
        approx_trace_dzdx = e_dzdx_e.view(
            x.shape[0], -1).sum(dim=1, keepdim=True)
        samples.append(approx_trace_dzdx)
    div = torch.cat(samples, dim=1).mean(dim=1, keepdim=True)

    if return_e_dzdx:
        return div, torch.stack(e_dzdx_vals, dim=1)
    else:
        return div

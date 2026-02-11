from typing import Union, Tuple, List
import torch


def delta_sq_norm_jac(e_dzdx: torch.Tensor):
    """

    :param torch.Tensor e_dzdx: tensor with shape `(batch_size, n_samples, event_size)`.
    :rtype: torch.Tensor.
    :return: delta of the squared Jacobian norm with shape `(batch_size,)`.
    """
    event_dims = list(range(1, len(e_dzdx.shape)))
    return e_dzdx.pow(2).mean(dim=event_dims)


def approximate_divergence(dz: torch.Tensor,
                           x: torch.Tensor,
                           noise: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
                           return_e_dzdx: bool = False):
    """Compute divergence term for ODE integration (integrates to log determinant).
    Approximation divergence as the Hessian trace.

    :param torch.Tensor dz: time derivative tensor (dx/dt) with shape `(batch_size, *event_shape)`.
    :param torch.Tensor dx: delta of the state, tensor with shape `(batch_size, *event_shape)`.
    :param torch.Tensor noise: tuple of one or more Hutchinson noise samples, each sample has shape `(batch_size, event_shape)`.
    :param bool return_e_dzdx: if True, return intermediate e_dzdx terms, useful for e.g., the delta of the squared 
     Jacobian norm computation.
    :rtype: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor].
    :return: divergence term tensor with shape `(batch_size,)`. 
        If return_e_dzdx is True, also return a stacked list of those terms.
    """
    if isinstance(noise, torch.Tensor):
        noise = (noise,)  # Handle e being a single sample
    elif not isinstance(noise, tuple):
        raise ValueError(
            "Hutchinson noise must be passed in as a tuple or torch.Tensor."
        )
    if dz.shape != x.shape:
        raise ValueError(f"Shapes of dz and x must match, but got {dz.shape = }, {x.shape = }")
    for n in noise:
        if n.shape != x.shape:
            raise ValueError("Shape of noise must match shape of x")
    
    samples = []
    e_dzdx_vals = []
    for e_ in noise:
        # Flatten
        with torch.no_grad():
            e_dzdx = torch.autograd.grad(dz, x, e_, create_graph=False, retain_graph=True)[0]
        # e_dzdx = torch.autograd.grad(dz, x, e_, create_graph=True)[0]
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

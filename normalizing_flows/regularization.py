import torch


def reconstruction_error(x, xr, event_shape: torch.Size, coefficient: float = 1.0):
    """
    aka "Inverse consistency"

    x, xr have shapes (*batch_shape, *event_shape)
    """
    n_event_dims = len(event_shape)
    event_dims = tuple(range(len(x.shape) - n_event_dims, len(x.shape), 1))
    return coefficient * torch.mean(torch.linalg.norm(x - xr, dim=event_dims))  # Norm over event dims, take the mean

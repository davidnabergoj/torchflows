import torch


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]


def keepdims_event_mask(batch_shape: torch.Tensor, event_mask: torch.Tensor):
    # Modifies event_mask to retain batch dimensions
    modified_event_mask = event_mask[(None,) * len(batch_shape)]
    return modified_event_mask


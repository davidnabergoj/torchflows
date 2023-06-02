import torch


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]

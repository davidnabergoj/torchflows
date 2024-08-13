import torch


def log_softmax(x, dim):
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


def log_sigmoid(x):
    return -torch.nn.functional.softplus(-x)


def log_dot(m1, m2):
    """
    m1 = log(A) with shape (..., d0, d1)
    m2 = log(B) with shape (..., d1, d2)
    Computes log(AB)
    """
    return torch.logsumexp(m1[..., :, :, None] + m2[..., None, :, :], dim=-2)

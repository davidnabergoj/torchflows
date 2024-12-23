import torch

from torchflows.bijections.finite.matrix import HouseholderProductMatrix


def matmul_with_householder(a: torch.Tensor, q: HouseholderProductMatrix):
    product = a
    for i in range(len(q.v)):
        product = product - 2 * q.v[i][:, None] * torch.matmul(q.v[i][None], product)
    return product

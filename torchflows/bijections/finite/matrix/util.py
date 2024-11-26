import torch

from torchflows.bijections.finite.matrix import HouseholderOrthogonalMatrix


def matmul_with_householder(a: torch.Tensor, q: HouseholderOrthogonalMatrix):
    product = a
    for i in range(len(q.v)):
        product = product - 2 * q.v[i][:, None] * torch.matmul(q.v[i][None], product)
    return product

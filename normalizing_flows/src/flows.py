import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from normalizing_flows.src.bijections import Bijection


class Flow(nn.Module):
    def __init__(self, bijection: Bijection):
        super().__init__()
        self.register_buffer('loc', torch.zeros(*bijection.event_shape))
        self.register_buffer('covariance_matrix', torch.eye(*bijection.event_shape))
        self.register_module('bijection', bijection)

    @property
    def base(self):
        return torch.distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        if context is not None:
            assert context.shape[0] == x.shape[0]
        z, log_det = self.bijection.forward(x, context=context)
        log_base = self.base.log_prob(z)
        return log_base + log_det

    def sample(self, n: int, context: torch.Tensor = None):
        """
        If context given, sample n vectors for each context vector.
        Otherwise, sample n vectors.

        :param n:
        :param context:
        :return:
        """
        if context is not None:
            z = self.base.sample(sample_shape=torch.Size((n, len(context))))
            context = context[None].repeat(*[n, *([1] * len(context.shape))])  # Make context shape match z shape
            assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base.sample(sample_shape=torch.Size((n,)))
        x, _ = self.bijection.inverse(z, context=context)
        return x

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 50,
            lr: float = 0.01,
            batch_size: int = None,
            shuffle: bool = True):
        if batch_size is None:
            batch_size = len(x_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dataset = TensorDataset(x_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for _ in range(n_epochs):
            for batch_x, in data_loader:
                optimizer.zero_grad()
                loss = -self.log_prob(batch_x).mean()
                loss.backward()
                optimizer.step()

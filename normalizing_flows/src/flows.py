import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
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

    def sample(self, n: int, context: torch.Tensor = None, no_grad: bool = False, return_log_prob: bool = False):
        """
        If context given, sample n vectors for each context vector.
        Otherwise, sample n vectors.

        :param n:
        :param context:
        :param no_grad:
        :return:
        """
        if context is not None:
            z = self.base.sample(sample_shape=torch.Size((n, len(context))))
            context = context[None].repeat(*[n, *([1] * len(context.shape))])  # Make context shape match z shape
            assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base.sample(sample_shape=torch.Size((n,)))
        if no_grad:
            z = z.detach()
            with torch.no_grad():
                x, log_det = self.bijection.inverse(z, context=context)
        else:
            x, log_det = self.bijection.inverse(z, context=context)
        x = x.to(self.loc)

        if return_log_prob:
            log_prob = self.base.log_prob(z) + log_det
            return x, log_prob
        return x

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 50,
            lr: float = 0.01,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False):
        if batch_size is None:
            batch_size = len(x_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dataset = TensorDataset(x_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if show_progress:
            iterator = tqdm(range(n_epochs), desc='Fitting NF')
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            for batch_x, in data_loader:
                optimizer.zero_grad()
                loss = -self.log_prob(batch_x).mean()
                loss.backward()
                optimizer.step()

    def variational_fit(self,
                        target,
                        n_epochs: int = 10,
                        lr: float = 0.01,
                        n_samples: int = 1000):
        # target must have a .sample method that takes as input the batch shape
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        for i in range(n_epochs):
            x_train = target.sample((n_samples,)).to(self.loc.device)
            optimizer.zero_grad()
            loss = -self.log_prob(x_train).mean()
            loss.backward()
            optimizer.step()

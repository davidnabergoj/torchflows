from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.continuous.ddnf import DeepDiffeomorphicBijection
from normalizing_flows.regularization import reconstruction_error
from normalizing_flows.utils import flatten_event, get_batch_shape, unflatten_event


class Flow(nn.Module):
    """
    Normalizing flow class.

    This class represents a bijective transformation of a standard Gaussian distribution (the base distribution).
    A normalizing flow is itself a distribution which we can sample from or use it to compute the density of inputs.
    """
    def __init__(self, bijection: Bijection):
        """

        :param bijection: transformation component of the normalizing flow.
        """
        super().__init__()
        self.register_module('bijection', bijection)
        self.register_buffer('loc', torch.zeros(self.bijection.n_dim))
        self.register_buffer('covariance_matrix', torch.eye(self.bijection.n_dim))

    @property
    def base(self) -> torch.distributions.Distribution:
        """
        :return: base distribution of the normalizing flow.
        """
        return torch.distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

    def base_log_prob(self, z: torch.Tensor):
        """
        Compute the log probability of input z under the base distribution.

        :param z: input tensor.
        :return: log probability of the input tensor.
        """
        zf = flatten_event(z, self.bijection.event_shape)
        log_prob = self.base.log_prob(zf)
        return log_prob

    def base_sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]]):
        """
        Sample from the base distribution.

        :param sample_shape: desired shape of sampled tensor.
        :return: tensor with shape sample_shape.
        """
        z_flat = self.base.sample(sample_shape)
        z = unflatten_event(z_flat, self.bijection.event_shape)
        return z

    def forward_with_log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Transform the input x to the space of the base distribution.

        :param x: input tensor.
        :param context: context tensor upon which the transformation is conditioned.
        :return: transformed tensor and the logarithm of the absolute value of the Jacobian determinant of the
         transformation.
        """
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)
        z, log_det = self.bijection.forward(x.to(self.loc), context=context)
        log_base = self.base_log_prob(z)
        return z, log_base + log_det

    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Compute the logarithm of the probability density of input x according to the normalizing flow.

        :param x: input tensor.
        :param context: context tensor.
        :return:
        """
        return self.forward_with_log_prob(x, context)[1]

    def sample(self, n: int, context: torch.Tensor = None, no_grad: bool = False, return_log_prob: bool = False):
        """
        Sample from the normalizing flow.

        If context given, sample n tensors for each context tensor.
        Otherwise, sample n tensors.

        :param n: number of tensors to sample.
        :param context: context tensor with shape c.
        :param no_grad: if True, do not track gradients in the inverse pass.
        :return: samples with shape (n, *event_shape) if no context given or (n, *c, *event_shape) if context given.
        """
        if context is not None:
            z = self.base_sample(sample_shape=torch.Size((n, len(context))))
            context = context[None].repeat(*[n, *([1] * len(context.shape))])  # Make context shape match z shape
            assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base_sample(sample_shape=torch.Size((n,)))
        if no_grad:
            z = z.detach()
            with torch.no_grad():
                x, log_det = self.bijection.inverse(z, context=context)
        else:
            x, log_det = self.bijection.inverse(z, context=context)
        x = x.to(self.loc)

        if return_log_prob:
            log_prob = self.base_log_prob(z) + log_det
            return x, log_prob
        return x

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None):
        """

        :param x_train:
        :param n_epochs:
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size:
        :param shuffle:
        :param show_progress:
        :param w_train: training data weights
        :return:
        """
        if w_train is None:
            batch_shape = get_batch_shape(x_train, self.bijection.event_shape)
            w_train = torch.ones(batch_shape)
        if batch_size is None:
            batch_size = len(x_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dataset = TensorDataset(x_train, w_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        n_event_dims = int(torch.prod(torch.as_tensor(self.bijection.event_shape)))

        if show_progress:
            iterator = tqdm(range(n_epochs), desc='Fitting NF')
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            for batch_x, batch_w in data_loader:
                optimizer.zero_grad()

                log_prob = self.log_prob(batch_x.to(self.loc))  # TODO context!
                w = batch_w.to(self.loc)
                assert log_prob.shape == w.shape
                loss = -torch.mean(log_prob * w) / n_event_dims

                if hasattr(self.bijection, 'regularization'):
                    loss += self.bijection.regularization()

                loss.backward()
                optimizer.step()

                if show_progress:
                    iterator.set_postfix_str(f'Loss: {loss:.4f}')

    def variational_fit(self,
                        target,
                        n_epochs: int = 10,
                        lr: float = 0.01,
                        n_samples: int = 1000,
                        show_progress: bool = False):
        # target must have a .sample method that takes as input the batch shape
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if show_progress:
            iterator = tqdm(range(n_epochs), desc='Variational NF fit')
        else:
            iterator = range(n_epochs)
        for i in iterator:
            x_train = target.sample((n_samples,)).to(self.loc.device)  # TODO context!
            optimizer.zero_grad()
            loss = -self.log_prob(x_train).mean()
            loss.backward()
            optimizer.step()

            if show_progress:
                iterator.set_postfix_str(f'loss: {float(loss):.4f}')


class DDNF(Flow):
    """
    Deep diffeomorphic normalizing flow.

    Salman et al. Deep diffeomorphic normalizing flows (2018).
    """

    def __init__(self, event_shape: torch.Size, **kwargs):
        bijection = DeepDiffeomorphicBijection(event_shape=event_shape, **kwargs)
        super().__init__(bijection)

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None,
            rec_err_coef: float = 1.0):
        """

        :param x_train:
        :param n_epochs:
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size:
        :param shuffle:
        :param show_progress:
        :param w_train: training data weights
        :param rec_err_coef: reconstruction error regularization coefficient.
        :return:
        """
        if w_train is None:
            batch_shape = get_batch_shape(x_train, self.bijection.event_shape)
            w_train = torch.ones(batch_shape)
        if batch_size is None:
            batch_size = len(x_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dataset = TensorDataset(x_train, w_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        n_event_dims = int(torch.prod(torch.as_tensor(self.bijection.event_shape)))

        if show_progress:
            iterator = tqdm(range(n_epochs), desc='Fitting NF')
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            for batch_x, batch_w in data_loader:
                optimizer.zero_grad()

                z, log_prob = self.forward_with_log_prob(batch_x.to(self.loc))  # TODO context!
                w = batch_w.to(self.loc)
                assert log_prob.shape == w.shape
                loss = -torch.mean(log_prob * w) / n_event_dims

                if hasattr(self.bijection, 'regularization'):
                    # Always true for DeepDiffeomorphicBijection, but we keep it for clarity
                    loss += self.bijection.regularization()

                # Inverse consistency regularization
                x_reconstructed = self.bijection.inverse(z)
                loss += reconstruction_error(batch_x, x_reconstructed, self.bijection.event_shape, rec_err_coef)

                # Geodesic regularization

                loss.backward()
                optimizer.step()

                if show_progress:
                    iterator.set_postfix_str(f'Loss: {loss:.4f}')

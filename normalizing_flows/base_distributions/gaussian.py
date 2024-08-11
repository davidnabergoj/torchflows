import torch
import torch.nn as nn
import math

from normalizing_flows.utils import sum_except_batch


class DiagonalGaussian(torch.distributions.Distribution, nn.Module):
    def __init__(self,
                 loc: torch.Tensor,
                 scale: torch.Tensor,
                 trainable_loc: bool = False,
                 trainable_scale: bool = False):
        super().__init__(event_shape=loc.shape)
        self.log_2_pi = math.log(2 * math.pi)
        if trainable_loc:
            self.register_parameter('loc', nn.Parameter(loc))
        else:
            self.register_buffer('loc', loc)

        if trainable_scale:
            self.register_parameter('log_scale', nn.Parameter(torch.log(scale)))
        else:
            self.register_buffer('log_scale', torch.log(scale))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        noise = torch.randn(size=(*sample_shape, *self.event_shape)).to(self.loc)
        sample_shape_mask = [None for _ in range(len(sample_shape))]
        return self.loc[sample_shape_mask] + noise * self.scale[sample_shape_mask]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if len(value.shape) <= len(self.event_shape):
            raise ValueError("Incorrect input shape")
        sample_shape_mask = [None for _ in range(len(value.shape) - len(self.event_shape))]
        loc = self.loc[sample_shape_mask]
        scale = self.scale[sample_shape_mask]
        log_scale = self.log_scale[sample_shape_mask]
        elementwise_log_prob = -(0.5 * ((value - loc) / scale) ** 2 + 0.5 * self.log_2_pi + log_scale)
        return sum_except_batch(elementwise_log_prob, self.event_shape)


class DenseGaussian(torch.distributions.Distribution, nn.Module):
    def __init__(self,
                 loc: torch.Tensor,
                 cov: torch.Tensor,
                 trainable_loc: bool = False):
        super().__init__(event_shape=loc.shape)
        event_size = int(torch.prod(torch.as_tensor(self.event_shape)))
        if cov.shape != (event_size, event_size):
            raise ValueError("Incorrect covariance matrix shape")

        self.log_2_pi = math.log(2 * math.pi)
        if trainable_loc:
            self.register_parameter('loc', nn.Parameter(loc))
        else:
            self.register_buffer('loc', loc)

        cholesky = torch.cholesky(cov)
        inverse_cholesky = torch.inverse(cholesky)
        inverse_cov = inverse_cholesky.T @ inverse_cholesky

        self.register_buffer('cholesky', cholesky)
        self.register_buffer('inverse_cov', inverse_cov)
        self.constant = -torch.sum(torch.log(torch.diag(cholesky))) - event_size / 2 * self.log_2_pi

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        flat_noise = torch.randn(size=(*sample_shape, int(torch.prod(torch.as_tensor(self.event_shape))))).to(self.loc)
        sample_shape_mask = [None for _ in range(len(sample_shape))]
        loc = self.loc[sample_shape_mask]
        return loc + (self.cholesky @ flat_noise).view_as(loc)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Without the determinant component
        if len(value.shape) <= len(self.event_shape):
            raise ValueError("Incorrect input shape")
        sample_shape_mask = [None for _ in range(len(value.shape) - len(self.event_shape))]
        diff = value - self.loc[sample_shape_mask]
        return self.constant - 0.5 * torch.einsum('...i,ij,...j->...', diff, self.inverse_cov, diff)

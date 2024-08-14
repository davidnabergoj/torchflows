from typing import List

import torch
import torch.nn as nn

from torchflows.base_distributions.gaussian import DiagonalGaussian, DenseGaussian
from torchflows.utils import get_batch_shape


class Mixture(torch.distributions.Distribution, nn.Module):
    """
    Base mixture distribution class. Extends torch.distributions.Distribution and torch.nn.Module.
    """
    def __init__(self,
                 components: List[torch.distributions.Distribution],
                 weights: torch.Tensor = None):
        """
        Mixture constructor.

        :param List[torch.distributions.Distribution] components: list of distribution components.
        :param torch.Tensor weights: tensor of weights with shape `(n_components,)`.
        """
        if weights is None:
            weights = torch.ones(len(components)) / len(components)
        super().__init__(event_shape=components[0].event_shape, validate_args=False)
        self.register_buffer('log_weights', torch.log(weights))
        self.components = components
        self.categorical = torch.distributions.Categorical(probs=weights)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        categories = self.categorical.sample(sample_shape)
        outputs = torch.zeros(*sample_shape, *self.event_shape).to(self.log_weights)
        for i, component in enumerate(self.components):
            outputs[categories == i] = component.sample(sample_shape)[categories == i]
        return outputs

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # We are assuming all components are normalized
        value = value.to(self.log_weights)
        batch_shape = get_batch_shape(value, self.event_shape)
        log_probs = torch.zeros(*batch_shape, self.n_components).to(self.log_weights)
        for i in range(self.n_components):
            log_probs[..., i] = self.components[i].log_prob(value)
        sample_shape_mask = [None for _ in range(len(value.shape) - len(self.event_shape))]
        return torch.logsumexp(self.log_weights[sample_shape_mask] + log_probs, dim=-1)


class DiagonalGaussianMixture(Mixture):
    """
    Mixture distribution of diagonal Gaussians. Extends Mixture.
    """

    def __init__(self,
                 locs: torch.Tensor,
                 scales: torch.Tensor,
                 weights: torch.Tensor = None,
                 trainable_locs: bool = False,
                 trainable_scales: bool = False):
        """
        DiagonalGaussianMixture constructor.

        :param torch.Tensor locs: tensor of locations with shape `(n_components, event_size)`.
        :param torch.Tensor scales: tensor of scales with shape `(n_components, event_size)`.
        :param torch.Tensor weights: tensor of weights with shape `(n_components,)`.
        :param bool trainable_locs: if True, make locations trainable.
        :param bool trainable_scales: if True, make scales trainable.
        """
        n_components, *event_shape = locs.shape
        components = []
        for i in range(n_components):
            components.append(DiagonalGaussian(locs[i], scales[i], trainable_locs, trainable_scales))
        super().__init__(components, weights)


class DenseGaussianMixture(Mixture):
    def __init__(self,
                 locs: torch.Tensor,
                 covs: torch.Tensor,
                 weights: torch.Tensor = None,
                 trainable_locs: bool = False):
        """
        DenseGaussianMixture constructor. Extends Mixture.

        :param torch.Tensor locs: tensor of locations with shape `(n_components, event_size)`.
        :param torch.Tensor covs: tensor of covariance matrices with shape `(n_components, event_size, event_size)`.
        :param torch.Tensor weights: tensor of weights with shape `(n_components,)`.
        :param bool trainable_locs: if True, make locations trainable.
        """
        n_components, *event_shape = locs.shape
        components = []
        for i in range(n_components):
            components.append(DenseGaussian(locs[i], covs[i], trainable_locs))
        super().__init__(components, weights)

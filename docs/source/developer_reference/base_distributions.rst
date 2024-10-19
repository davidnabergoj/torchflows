Base distributions
==========================

Existing base distributions
-----------------------------

.. autoclass:: torchflows.base_distributions.gaussian.DiagonalGaussian
    :members: __init__

.. autoclass:: torchflows.base_distributions.gaussian.DenseGaussian
    :members: __init__

.. autoclass:: torchflows.base_distributions.mixture.DiagonalGaussianMixture
    :members: __init__

.. autoclass:: torchflows.base_distributions.mixture.DenseGaussianMixture
    :members: __init__

Creating new base distributions
-----------------------------------

To create a new base distribution, we must create a subclass of :class:`torch.distributions.Distribution` and :class:`torch.nn.Module`.
This class should support the methods sampling and log probability computation.
We give an example for the diagonal Gaussian base distribution:

.. code-block:: python

    import torch
    import torch.distributions
    import torch.nn as nn
    import math

    class DiagonalGaussian(torch.distributions.Distribution, nn.Module):
        def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
            super().__init__(event_shape=loc.shape, validate_args=False)
            self.log_2_pi = math.log(2 * math.pi)
            self.register_buffer('loc', loc)
            self.register_buffer('log_scale', torch.log(scale))

        @property
        def scale(self):
            return torch.exp(self.log_scale)

        def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            noise = torch.randn(size=(*sample_shape, *self.event_shape)).to(self.loc)
            # Unsqueeze loc and scale to match batch shape
            sample_shape_mask = [None for _ in range(len(sample_shape))]
            return self.loc[sample_shape_mask] + noise * self.scale[sample_shape_mask]

        def log_prob(self, value: torch.Tensor) -> torch.Tensor:
            if len(value.shape) <= len(self.event_shape):
                raise ValueError("Incorrect input shape")
            # Unsqueeze loc and scale to match batch shape
            sample_shape_mask = [None for _ in range(len(value.shape) - len(self.event_shape))]
            loc = self.loc[sample_shape_mask]
            scale = self.scale[sample_shape_mask]
            log_scale = self.log_scale[sample_shape_mask]

            # Compute log probability
            elementwise_log_prob = -(0.5 * ((value - loc) / scale) ** 2 + 0.5 * self.log_2_pi + log_scale)
            return sum_except_batch(elementwise_log_prob, self.event_shape)
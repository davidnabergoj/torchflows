Base distributions
==========================

Existing base distributions
-----------------------------

.. autoclass:: torchflows.base_distributions.gaussian.DiagonalGaussian
    :members: __init__

.. autoclass:: torchflows.base_distributions.gaussian.StandardGaussian
    :members: __init__

.. autoclass:: torchflows.base_distributions.gaussian.DenseGaussian
    :members: __init__

.. autoclass:: torchflows.base_distributions.cauchy.ZeroCauchy
    :members: __init__

.. autoclass:: torchflows.base_distributions.mixture.Mixture
    :members: __init__

.. autoclass:: torchflows.base_distributions.mixture.DiagonalGaussianMixture
    :members: __init__

.. autoclass:: torchflows.base_distributions.mixture.DenseGaussianMixture
    :members: __init__

.. autoclass:: torchflows.base_distributions.concatenation.ConcatenatedDistribution
    :members: __init__

``ConcatenatedDistribution`` joins distributions over disjoint parts of the (flattened) event.
For example, a Gaussian over the first two dimensions and a Cauchy over the last three:

.. code-block:: python

    import torch
    from torchflows.base_distributions.concatenation import ConcatenatedDistribution
    from torchflows.base_distributions.gaussian import StandardGaussian
    from torchflows.base_distributions.cauchy import ZeroCauchy

    base = ConcatenatedDistribution([StandardGaussian((2,)), ZeroCauchy((3,), scale=1.0)])
    z = base.sample((10,))
    print(z.shape)  # (10, 5)
    print(base.log_prob(z).shape)  # (10,)

Creating new base distributions
-----------------------------------

To create a new base distribution, we must create a subclass of :class:`torch.distributions.Distribution` and :class:`torch.nn.Module`.
This class should support the methods sampling and log probability computation.
The event shape of the base distribution should be ``(event_size,)``, since the flow passes flattened events to it.
We give an example for the diagonal Gaussian base distribution:

.. code-block:: python

    import math
    import torch
    import torch.distributions
    import torch.nn as nn
    from torchflows.utils import sum_except_batch

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

The new distribution is then passed to the flow:

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    base = DiagonalGaussian(loc=torch.zeros(4), scale=torch.ones(4))
    flow = Flow(RealNVP(event_shape=(4,)), base_distribution=base)
    print(flow.sample(3).shape)  # (3, 4)

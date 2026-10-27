Choosing a base distribution
==============================

We may replace the default standard Gaussian distribution with any torch distribution that is also a module.
Some custom distributions are already implemented in :mod:`torchflows.base_distributions`.
We show an example for a diagonal Gaussian base distribution with mean 3 and standard deviation 2.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP
    from torchflows.base_distributions.gaussian import DiagonalGaussian

    torch.manual_seed(0)
    event_shape = (10,)
    base_distribution = DiagonalGaussian(
        loc=torch.full(size=event_shape, fill_value=3.0),
        scale=torch.full(size=event_shape, fill_value=2.0),
    )
    flow = Flow(RealNVP(event_shape), base_distribution=base_distribution)

    x_new = flow.sample((10,))

Heavy-tailed base distributions
---------------------------------

A Cauchy base distribution can help model heavy-tailed data.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP
    from torchflows.base_distributions.cauchy import ZeroCauchy

    torch.manual_seed(0)
    event_shape = (10,)
    flow = Flow(RealNVP(event_shape), base_distribution=ZeroCauchy(event_shape, scale=1.0))

    x_new = flow.sample((10,))

Mixture base distributions
---------------------------------

Gaussian mixtures are suitable for multimodal data.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP
    from torchflows.base_distributions.mixture import DiagonalGaussianMixture

    torch.manual_seed(0)
    event_shape = (10,)
    n_components = 3
    base_distribution = DiagonalGaussianMixture(
        locs=torch.randn(size=(n_components, *event_shape)) * 3,  # one location per component
        scales=torch.ones(size=(n_components, *event_shape)),
        weights=torch.tensor([0.5, 0.3, 0.2])
    )
    flow = Flow(RealNVP(event_shape), base_distribution=base_distribution)

    x_new = flow.sample((10,))

Nontrivial event shapes
------------------------

When the event has more than one axis, the base distribution must deal with flattened data. We show an example below.

.. note::

    The requirement to work with flattened data may change in the future.


.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP
    from torchflows.base_distributions.gaussian import DiagonalGaussian

    torch.manual_seed(0)
    event_shape = (2, 3, 5)
    event_size = int(torch.prod(torch.as_tensor(event_shape)))
    base_distribution = DiagonalGaussian(
        loc=torch.full(size=(event_size,), fill_value=3.0),
        scale=torch.full(size=(event_size,), fill_value=2.0),
    )
    flow = Flow(RealNVP(event_shape), base_distribution=base_distribution)

    x_new = flow.sample((10,))
    print(x_new.shape)  # (10, 2, 3, 5)

See the :doc:`base distribution reference <../developer_reference/base_distributions>` for all available distributions and to learn how to create new ones.

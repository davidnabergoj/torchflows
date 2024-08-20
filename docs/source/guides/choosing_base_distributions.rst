Choosing a base distribution
==============================

We may replace the default standard Gaussian distribution with any torch distribution that is also a module.
Some custom distributions are already implemented.
We show an example for a diagonal Gaussian base distribution with mean 3 and standard deviation 2.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP
    from torchflows.base_distributions.gaussian import DiagonalGaussian

    torch.manual_seed(0)
    event_shape = (10,)
    base_distribution = DiagonalGaussian(
        loc=torch.full(size=event_shape, fill_value=3.0),
        scale=torch.full(size=event_shape, fill_value=2.0),
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
    from torchflows.architectures import RealNVP
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

Basic usage
==============

All Torchflow models are constructed as a combination of a bijection and a base distribution.
Both the bijection and base distribution objects work on events (tensors) with a set event shape.
A bijection and a distribution instance are are packaged together into a `Flow` object, creating a trainable torch module.
The simplest way to create a normalizing flow is to import an existing architecture and wrap it with a `Flow` object.
In the example below, we use the Real NVP architecture.
We do not specify a base distribution, so the default standard Gaussian is chosen.

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP

    event_shape = (10,)  # suppose our data are 10-dimensional vectors
    flow = Flow(RealNVP(event_shape))

Normalizing flows learn the distributions of unlabeled data.
We provide an example on how to train a flow for a dataset of 50-dimensional vectors.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP

    torch.manual_seed(0)

    n_data = 1000
    n_dim = 50

    x = torch.randn(n_data, n_dim)  # Generate synthetic training data
    flow = Flow(RealNVP(n_dim))  # Create the normalizing flow
    flow.fit(x, show_progress=True)  # Fit the normalizing flow to training data

After fitting the flow, we can use it to sample new data and compute the log probability density of data points.

.. code-block:: python

    x_new = flow.sample(50)  # Sample 50 new data points
    print(x_new.shape)  # (50, 3)

    log_prob = flow.log_prob(x)  # Compute the data log probability
    print(log_prob.shape)  # (100,)

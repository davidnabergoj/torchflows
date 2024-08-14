Basic usage
==============

Torchflow models learn the distributions of unlabeled data. We provide an example on how to train a normalizing flow for a dataset of 50-dimensional vectors.

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

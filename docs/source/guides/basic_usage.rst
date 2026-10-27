Basic usage
==============

All Torchflows models are constructed as a combination of a bijection and a base distribution.
Both the bijection and base distribution objects work on events (tensors) with a set event shape.
A bijection and a distribution instance are packaged together into a ``Flow`` object, creating a trainable torch module.
The simplest way to create a normalizing flow is to import an existing architecture and wrap it with a ``Flow`` object.
In the example below, we use the Real NVP architecture.
We do not specify a base distribution, so the default standard Gaussian is chosen.

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    event_shape = (10,)  # suppose our data are 10-dimensional vectors
    flow = Flow(RealNVP(event_shape))

Architecture presets are grouped by family:

* autoregressive presets in :mod:`torchflows.bijections.finite.autoregressive` (e.g., ``RealNVP``, ``MAF``, ``CouplingRQNSF``);
* multiscale presets for images in :mod:`torchflows.bijections.finite.multiscale` (e.g., ``MultiscaleRealNVP``, ``AffineGlow``);
* residual presets in :mod:`torchflows.bijections.finite.residual` (e.g., ``ResFlow``, ``InvertibleResNet``);
* continuous presets in :mod:`torchflows.bijections.continuous` (e.g., ``FFJORD``, ``RNODE``, ``OTFlowBijection``).

See the :doc:`full list of architectures <../architectures/index>`.

Fitting a flow to data
-------------------------

Normalizing flows learn the distributions of unlabeled data.
We provide an example on how to train a flow for a dataset of 50-dimensional vectors.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)

    n_data = 1000
    n_dim = 50

    x = torch.randn(n_data, n_dim)  # Generate synthetic training data
    flow = Flow(RealNVP(event_shape=(n_dim,)))  # Create the normalizing flow
    flow.fit(x, show_progress=True)  # Fit the normalizing flow to training data

After fitting the flow, we can use it to sample new data and compute the log probability density of data points.

.. code-block:: python

    x_new = flow.sample(50)  # Sample 50 new data points
    print(x_new.shape)  # (50, 50)

    log_prob = flow.log_prob(x)  # Compute the data log probability
    print(log_prob.shape)  # (1000,)

    # Sample and compute the log probability of the samples in a single pass
    x_new, log_prob_new = flow.sample(50, return_log_prob=True)
    print(log_prob_new.shape)  # (50,)

    # Sample without tracking gradients (faster, less memory)
    x_new = flow.sample(50, no_grad=True)

Validation data and early stopping
-----------------------------------------

The ``fit`` method accepts validation data.
By default, the weights with the lowest validation loss are restored after training (``keep_best_weights=True``).
We can also stop training once the validation loss stops improving, or set a time limit.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    x = torch.randn(size=(1200, 5))
    x_train, x_val = x[:1000], x[1000:]

    flow = Flow(RealNVP(event_shape=(5,)))
    flow.fit(
        x_train,
        x_val=x_val,
        n_epochs=1000,
        lr=0.01,
        batch_size=256,
        early_stopping=True,
        early_stopping_threshold=20,  # stop after 20 epochs without improvement
        time_limit_seconds=60,
        show_progress=True
    )

Weighted data
-------------------------

Each training point can be assigned a weight with ``w_train`` (and ``w_val`` for validation data).
Weights scale the contribution of each point to the loss.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    x_train = torch.randn(size=(1000, 3))
    w_train = torch.rand(size=(1000,))

    flow = Flow(RealNVP(event_shape=(3,)))
    flow.fit(x_train, w_train=w_train, n_epochs=100)

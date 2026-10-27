Event shapes
======================

Torchflows supports modeling tensors with arbitrary shapes.
The event shape is passed to the bijection constructor as a tuple.
For example, we can model events with shape ``(2, 3, 5)`` as follows:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    event_shape = (2, 3, 5)
    n_data = 1000
    x_train = torch.randn(size=(n_data, *event_shape))
    print(x_train.shape)  # (1000, 2, 3, 5)

    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, show_progress=True)

    x_new = flow.sample((500,))
    print(x_new.shape)  # (500, 2, 3, 5)

Batch shapes can also have more than one axis.
All leading axes before the event axes are treated as batch axes:

.. code-block:: python

    x_new = flow.sample((4, 7))
    print(x_new.shape)  # (4, 7, 2, 3, 5)

    log_prob = flow.log_prob(x_new)
    print(log_prob.shape)  # (4, 7)

.. note::

    Always pass the event shape as a tuple, e.g., ``(10,)`` instead of ``10``.
    Some autoregressive presets also accept integers, but residual and continuous bijections do not.

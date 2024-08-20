Event shapes
======================

Torchflows supports modeling tensors with arbitrary shapes. For example, we can model events with shape `(2, 3, 5)` as follows:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP

    torch.manual_seed(0)
    event_shape = (2, 3, 5)
    n_data = 1000
    x_train = torch.randn(size=(n_data, *event_shape))
    print(x_train.shape)  # (1000, 2, 3, 5)

    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, show_progress=True)

    x_new = flow.sample((500,))
    print(x_new.shape)  # (500, 2, 3, 5)
Using CUDA
===========

Torchflows models are torch modules and thus seamlessly support CUDA (and other devices).
When using the ``fit`` method, training data is automatically transferred onto the flow device.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    event_shape = (10,)
    x_train = torch.randn(size=(1000, *event_shape))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow = Flow(RealNVP(event_shape)).to(device)
    flow.fit(x_train, show_progress=True)

    x_new = flow.sample(100)  # sampled on the flow device
    print(x_new.device)

The ``log_prob`` method also moves inputs onto the flow device.

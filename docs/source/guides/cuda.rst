Using CUDA
===========

Torchflows models are torch modules and thus seamlessly support CUDA (and other devices).
When using the *fit* method, training data is automatically transferred onto the flow device.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP

    torch.manual_seed(0)
    event_shape = (10,)
    x_train = torch.randn(size=(1000, *event_shape))

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train, show_progress=True)
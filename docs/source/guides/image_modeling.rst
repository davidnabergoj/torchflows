Image modeling
==============

When modeling images, we can use specialized multiscale architectures which use convolutional neural network conditioners and specialized coupling schemes.
These architectures expect event shapes to be *(channels, height, width)*.

.. note::
    Multiscale architectures are currently undergoing improvements.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import MultiscaleRealNVP

    image_shape = (3, 28, 28)
    n_images = 100

    torch.manual_seed(0)
    training_images = torch.randn(size=(n_images, *image_shape))  # synthetic data
    flow = Flow(MultiscaleRealNVP(image_shape))
    flow.fit(training_images, show_progress=True)
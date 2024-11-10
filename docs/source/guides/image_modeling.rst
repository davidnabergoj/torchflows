Image modeling
=================

When modeling images, we can use specialized multiscale architectures which use convolutional neural network conditioners and specialized coupling schemes.
These architectures expect event shapes to be *(channels, height, width)*.
See the :ref:`list of multiscale architecture presets here <multiscale_architecture_list>`.

Basic multiscale architectures
---------------------------------------
We provide some basic multiscale presets and give an example for the RealNVP variant below:

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

Glow-style multiscale architectures
-------------------------------------------

Glow-style architectures are extensions of basic multiscale architectures which use an additional invertible 1x1 convolution in each layer.
We give an example for Glow with affine transformers below:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import AffineGlow

    image_shape = (3, 28, 28)
    n_images = 100

    torch.manual_seed(0)
    training_images = torch.randn(size=(n_images, *image_shape))  # synthetic data
    flow = Flow(AffineGlow(image_shape))
    flow.fit(training_images, show_progress=True)
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
    from torchflows.bijections.finite.multiscale import MultiscaleRealNVP

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
    from torchflows.bijections.finite.multiscale import AffineGlow

    image_shape = (3, 28, 28)
    n_images = 100

    torch.manual_seed(0)
    training_images = torch.randn(size=(n_images, *image_shape))  # synthetic data
    flow = Flow(AffineGlow(image_shape))
    flow.fit(training_images, show_progress=True)

Convolutional residual and continuous architectures
-------------------------------------------------------

Residual and continuous families also provide convolutional presets for images.
Continuous bijections are wrapped with :class:`~torchflows.flows.ContinuousFlow` (see :doc:`continuous_flows`).

.. code-block:: python

    import torch
    from torchflows.flows import Flow, ContinuousFlow
    from torchflows.bijections.finite.residual import ConvolutionalResFlow
    from torchflows.bijections.continuous import ConvolutionalFFJORD

    image_shape = (3, 8, 8)
    torch.manual_seed(0)
    training_images = torch.randn(size=(50, *image_shape))  # synthetic data

    residual_flow = Flow(ConvolutionalResFlow(image_shape))
    residual_flow.fit(training_images, n_epochs=10)

    continuous_flow = ContinuousFlow(ConvolutionalFFJORD(image_shape))
    continuous_flow.fit(training_images, n_epochs=10)

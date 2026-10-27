API for multiscale architectures
========================================================

Multiscale architectures are suitable for image modeling.
Import them from :mod:`torchflows.bijections.finite.multiscale`, e.g., ``from torchflows.bijections.finite.multiscale import AffineGlow``.

.. _multiscale_architecture_api:


Classic multiscale architectures
----------------------------------

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleNICE
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleRealNVP
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleRQNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleLRSNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepSigmoid
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDenseSigmoid
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepDenseSigmoid
    :members: __init__


Glow-style multiscale architectures
-------------------------------------

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.AffineGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.ShiftGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.RQSGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.LRSGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.DeepSigmoidGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.DenseSigmoidGlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.DeepDenseSigmoidGlow
    :members: __init__


Convolutional residual architectures
---------------------------------------

.. autoclass:: torchflows.bijections.finite.residual.architectures.ConvolutionalInvertibleResNet
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.ConvolutionalResFlow
    :members: __init__


Convolutional continuous architectures
---------------------------------------

.. autoclass:: torchflows.bijections.continuous.ddnf.ConvolutionalDDNF
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.ffjord.ConvolutionalFFJORD
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.rnode.ConvolutionalRNODE
    :members: __init__

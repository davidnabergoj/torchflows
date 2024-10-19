API for multiscale architectures
========================================================

Multiscale architectures are suitable for image modeling.

.. _multiscale_architecture_api:


Classic multiscale architectures
------------------------------

.. autoclass:: torchflows.architectures.MultiscaleNICE
    :members: __init__

.. autoclass:: torchflows.architectures.MultiscaleRealNVP
    :members: __init__

.. autoclass:: torchflows.architectures.MultiscaleRQNSF
    :members: __init__

.. autoclass:: torchflows.architectures.MultiscaleLRSNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepSigmoid
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDenseSigmoid
    :members: __init__

.. autoclass:: torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepDenseSigmoid
    :members: __init__


Glow-style multiscale architectures
------------------------------

.. autoclass:: torchflows.architectures.AffineGlow
    :members: __init__

.. autoclass:: torchflows.architectures.ShiftGlow
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

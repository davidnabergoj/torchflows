API for standard architectures
============================
We lists notable implemented bijection architectures.
These all inherit from the Bijection class.

.. _autoregressive_architecture_api:

Autoregressive architectures
--------------------------------

.. autoclass:: torchflows.architectures.RealNVP
    :members: __init__

.. autoclass:: torchflows.architectures.InverseRealNVP
    :members: __init__

.. autoclass:: torchflows.architectures.NICE
    :members: __init__

.. autoclass:: torchflows.architectures.MAF
    :members: __init__

.. autoclass:: torchflows.architectures.IAF
    :members: __init__

.. autoclass:: torchflows.architectures.CouplingRQNSF
    :members: __init__

.. autoclass:: torchflows.architectures.MaskedAutoregressiveRQNSF
    :members: __init__

.. autoclass:: torchflows.architectures.InverseAutoregressiveRQNSF
    :members: __init__

.. autoclass:: torchflows.architectures.CouplingLRS
    :members: __init__

.. autoclass:: torchflows.architectures.MaskedAutoregressiveLRS
    :members: __init__

.. autoclass:: torchflows.architectures.InverseAutoregressiveLRS
    :members: __init__

.. autoclass:: torchflows.architectures.CouplingDSF
    :members: __init__

.. autoclass:: torchflows.architectures.UMNNMAF
    :members: __init__

.. _continuous_architecture_api:

Continuous architectures
-------------------------
.. autoclass:: torchflows.architectures.DeepDiffeomorphicBijection
    :members: __init__

.. autoclass:: torchflows.architectures.RNODE
    :members: __init__

.. autoclass:: torchflows.architectures.FFJORD
    :members: __init__

.. autoclass:: torchflows.architectures.OTFlow
    :members: __init__

.. _residual_architecture_api:

Residual architectures
-----------------------
.. autoclass:: torchflows.architectures.ResFlow
    :members: __init__

.. autoclass:: torchflows.architectures.ProximalResFlow
    :members: __init__

.. autoclass:: torchflows.architectures.InvertibleResNet
    :members: __init__

.. autoclass:: torchflows.architectures.PlanarFlow
    :members: __init__

.. autoclass:: torchflows.architectures.RadialFlow
    :members: __init__

.. autoclass:: torchflows.architectures.SylvesterFlow
    :members: __init__

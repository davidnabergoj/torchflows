Standard architectures
============================
We lists notable implemented bijection architectures.
These all inherit from the Bijection class.

.. _architectures:

Autoregressive architectures
--------------------------------

.. autoclass:: torchflows.architectures.RealNVP
.. autoclass:: torchflows.architectures.InverseRealNVP
.. autoclass:: torchflows.architectures.NICE
.. autoclass:: torchflows.architectures.MAF
.. autoclass:: torchflows.architectures.IAF
.. autoclass:: torchflows.architectures.CouplingRQNSF
.. autoclass:: torchflows.architectures.MaskedAutoregressiveRQNSF
.. autoclass:: torchflows.architectures.InverseAutoregressiveRQNSF
.. autoclass:: torchflows.architectures.CouplingLRS
.. autoclass:: torchflows.architectures.MaskedAutoregressiveLRS
.. autoclass:: torchflows.architectures.CouplingDSF
.. autoclass:: torchflows.architectures.UMNNMAF

Continuous architectures
-------------------------
.. autoclass:: torchflows.architectures.DeepDiffeomorphicBijection
.. autoclass:: torchflows.architectures.RNODE
.. autoclass:: torchflows.architectures.FFJORD
.. autoclass:: torchflows.architectures.OTFlow

Residual architectures
-----------------------
.. autoclass:: torchflows.architectures.ResFlow
.. autoclass:: torchflows.architectures.ProximalResFlow
.. autoclass:: torchflows.architectures.InvertibleResNet
.. autoclass:: torchflows.architectures.PlanarFlow
.. autoclass:: torchflows.architectures.RadialFlow
.. autoclass:: torchflows.architectures.SylvesterFlow
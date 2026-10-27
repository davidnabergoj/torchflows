API for standard architectures
=================================================
We list notable implemented bijection architectures.
These all inherit from the Bijection class.

.. _autoregressive_architecture_api:

Autoregressive architectures
--------------------------------
Import from :mod:`torchflows.bijections.finite.autoregressive`, e.g., ``from torchflows.bijections.finite.autoregressive import RealNVP``.

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.RealNVP
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseRealNVP
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.NICE
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MAF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.IAF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.CouplingRQNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveRQNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveRQNSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.CouplingLRS
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveLRS
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveLRS
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.CouplingDeepSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.CouplingDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.CouplingDeepDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepDenseSF
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.architectures.UMNNMAF
    :members: __init__

.. _continuous_architecture_api:

Continuous architectures
-------------------------
Import from :mod:`torchflows.bijections.continuous` and wrap with :class:`~torchflows.flows.ContinuousFlow`.

.. autoclass:: torchflows.bijections.continuous.ddnf.DDNF
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.rnode.RNODE
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.ffjord.FFJORD
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.otflow.OTFlowBijection
    :members: __init__

.. _residual_architecture_api:

Residual architectures
-----------------------
Import from :mod:`torchflows.bijections.finite.residual`.

.. autoclass:: torchflows.bijections.finite.residual.architectures.ResFlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.ProximalResFlow
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.InvertibleResNet
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.PlanarBijection
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.RadialBijection
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.architectures.SylvesterBijection
    :members: __init__

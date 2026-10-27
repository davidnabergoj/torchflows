List of conditioner transforms
==============================

Conditioner transforms predict transformer parameters from the conditioning part of the input (and optional context).
Pass a conditioner transform class to an autoregressive layer or architecture with ``conditioner_transform_class`` and its arguments with ``conditioner_kwargs``.

.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.FeedForward
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.Linear
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.ResidualFeedForward
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.Constant
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.MADE
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.LinearMADE

Example:

.. code-block:: python

    import torch.nn as nn
    from torchflows.bijections.finite.autoregressive.layers import AffineCoupling
    from torchflows.bijections.finite.autoregressive.conditioning.transforms import ResidualFeedForward

    layer = AffineCoupling(
        event_shape=(10,),
        conditioner_transform_class=ResidualFeedForward,
        conditioner_kwargs={'n_layers': 3, 'n_hidden': 32, 'nonlinearity': nn.ReLU}
    )

Conditioner combinations
--------------------------
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.CombinedConditioner

Base classes
---------------
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.ConditionerTransform
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.ElementwiseConditionerTransform
.. autoclass:: torchflows.bijections.finite.autoregressive.conditioning.transforms.TensorConditionerTransform

Bijections
====================

All normalizing flow transformations are bijections and compositions thereof.

Base bijections
------------------

The following classes define forward and inverse pass methods which all bijections inherit.
Both methods return the transformed tensor and the log absolute value of the Jacobian determinant.

.. autoclass:: torchflows.bijections.base.Bijection
    :members: __init__, forward, inverse

.. autoclass:: torchflows.bijections.base.BijectiveComposition
    :members: __init__

A composition receives a list of bijections and infers the event shape from the first one:

.. code-block:: python

    import torch
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive.layers import AffineCoupling, ActNorm
    from torchflows.bijections.finite.matrix import ReversePermutationMatrix

    event_shape = (6,)
    bijection = BijectiveComposition([
        AffineCoupling(event_shape),
        ReversePermutationMatrix(event_shape),
        AffineCoupling(event_shape),
        ActNorm(event_shape),
    ])

    x = torch.randn(size=(100, *event_shape))
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)
    print(z.shape, log_det_forward.shape)  # (100, 6) (100,)

Creating a new bijection
------------------------------

A new bijection subclasses :class:`~torchflows.bijections.base.Bijection` and implements ``forward`` and ``inverse``.
Below is an elementwise exponential scaling with a trainable log scale:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchflows.bijections.base import Bijection
    from torchflows.flows import Flow
    from torchflows.utils import get_batch_shape

    class ElementwiseExpScale(Bijection):
        def __init__(self, event_shape, **kwargs):
            super().__init__(event_shape, **kwargs)
            self.log_scale = nn.Parameter(torch.zeros(size=event_shape))

        def log_det(self, batch_shape):
            return self.log_scale.sum() * torch.ones(size=batch_shape, device=self.log_scale.device)

        def forward(self, x, context=None):
            batch_shape = get_batch_shape(x, self.event_shape)
            return x * torch.exp(self.log_scale), self.log_det(batch_shape)

        def inverse(self, z, context=None):
            batch_shape = get_batch_shape(z, self.event_shape)
            return z * torch.exp(-self.log_scale), -self.log_det(batch_shape)

    flow = Flow(ElementwiseExpScale(event_shape=(3,)))
    flow.fit(torch.randn(size=(100, 3)) * 5, n_epochs=50)

Bijection subclasses for different NF families
------------------------------------------------------------------
To improve efficiency of forward and inverse passes in NF layers, we subclass the base bijections with respect to different families of NF architectures.
On the pages below, we list base classes for each family, and provide a list of already implemented classes.

.. toctree::
    :maxdepth: 1

    autoregressive/bijections
    residual/bijections
    continuous/bijections
    matrix

Inverting a bijection
------------------------------

Each bijection can be inverted with the ``invert`` function.
For example, inverting MAF gives a flow with fast sampling and slow density evaluation.

.. autofunction:: torchflows.bijections.base.invert

.. code-block:: python

    from torchflows.bijections.base import invert
    from torchflows.bijections.finite.autoregressive import MAF

    inverse_maf = invert(MAF(event_shape=(5,)))

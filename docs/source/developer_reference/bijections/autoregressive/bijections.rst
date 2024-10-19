Autoregressive bijections
===========================

Autoregressive bijections belong in one of two categories: coupling or masked autoregressive bijections.
Architectures like IAF make use of the inverse masked autoregressive bijection, which simply swaps the `forward` and `inverse` methods of its corresponding masked autoregressive counterpart.
Multiscale architectures are special cases of coupling architectures.
Each autoregressive bijection consists of a transformer (parameterized bijection that transforms a part of the input), a conditioner, and a conditioner transform (model that predicts transformer parameters).
See the :doc:`transformers` and :doc:`conditioner_transforms` sections for more details.
To improve performance, we define subclasses according to the conditioner type.
We list these subclasses in the rest of the document.

Coupling bijections
--------------------------------------------------

Coupling architectures are compositions of coupling bijections, which extend the following base class:

.. autoclass:: torchflows.bijections.finite.autoregressive.layers_base.CouplingBijection
    :members: __init__

We give an example on how to create a custom coupling bijection using a transformer, coupling strategy, and conditioner transform:

.. code-block:: python

    from torchflows.bijections.finite.autoregressive.layers_base import CouplingBijection
    from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Affine
    from torchflows.bijections.finite.autoregressive.conditioning.coupling_masks import HalfSplit
    from torchflows.bijections.finite.autoregressive.conditioning.transforms import ResidualFeedForward

    class AffineCoupling(CouplingBijection):
        def __init__(self, event_shape, **kwargs):
            coupling = HalfSplit(event_shape)
            super().__init__(
                event_shape,
                transformer_class=Affine,
                coupling=HalfSplit(event_shape),
                conditioner_transform_class=ResidualFeedForward
            )

    event_shape = (10,)  # say we have vectors of size 10
    bijection = AffineCoupling(event_shape)  # create the bijection

Masked autoregressive bijections
----------------------------------------

Masked autoregressive and inverse autoregressive architectures are compositions of their respective bijections, extending one of the following classes:

.. autoclass:: torchflows.bijections.finite.autoregressive.layers_base.MaskedAutoregressiveBijection
    :members: __init__

.. autoclass:: torchflows.bijections.finite.autoregressive.layers_base.InverseMaskedAutoregressiveBijection
    :members: __init__

We give an example on how to create a custom coupling bijection using a transformer, coupling strategy, and conditioner transform:

.. code-block:: python

    from torchflows.bijections.finite.autoregressive.layers_base import CouplingBijection
    from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Affine
    from torchflows.bijections.finite.autoregressive.conditioning.transforms import ResidualFeedForward

    class AffineForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
        def __init__(self, event_shape, **kwargs):
            super().__init__(
                event_shape,
                transformer_class=Affine,
                conditioner_transform_class=ResidualFeedForward
            )

    class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
        def __init__(self, event_shape, **kwargs):
            super().__init__(
                event_shape,
                transformer_class=Affine,
                conditioner_transform_class=ResidualFeedForward
            )


    # say we have 100 vectors of size 10
    event_shape = (10,)
    x = torch.randn(size=(100, *event_shape))

    bijection = AffineCoupling(event_shape)  # create the bijection
    z, log_det_forward = bijection.forward(x)
    y, log_det_inverse = bijection.inverse(z)


Multiscale autoregressive bijections
--------------------------------------------------

Multiscale architectures are coupling architectures which are specialized for image modeling, extending the class below:

.. autoclass:: torchflows.bijections.finite.multiscale.base.MultiscaleBijection
    :members: __init__

See also
------------

.. toctree::
    :maxdepth: 1

    transformers
    conditioners
    conditioner_transforms
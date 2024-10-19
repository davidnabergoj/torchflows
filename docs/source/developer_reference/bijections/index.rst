Bijections
====================

All normalizing flow transformations are bijections and compositions thereof.

Base bijections
------------------

The following classes define forward and inverse pass methods which all bijections inherit.

.. autoclass:: torchflows.bijections.base.Bijection
    :members: __init__, forward, inverse

.. autoclass:: torchflows.bijections.base.BijectiveComposition
    :members: __init__


Bijection subclasses for different NF families
------------------------------------------------------------------
To improve efficiency of forward and inverse passes in NF layers, we subclass the base bijections with respect to different families of NF architectures.
On the pages below, we list base classes for each family, and provide a list of already implemented classes.

.. toctree::
    :maxdepth: 1

    autoregressive/bijections
    residual/bijections
    continuous/bijections

Inverting a bijection
------------------------------

Each bijection can be inverted with the `invert` function.

.. autofunction:: torchflows.bijections.base.invert
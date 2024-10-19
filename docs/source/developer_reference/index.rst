Developer reference
===========================

This section describes how to create NF architectures and NF components in Torchflows.
NFs consist of two main components:

* a base distribution,
* a bijection.

In Torchflows, we further wrap these two with the :class:`torchflows.flows.Flow` object or one of its subclasses to enable e.g., fitting NFs, computing the log probability density, and sampling.

At its core, each of these components is a PyTorch module which extends existing base classes:

* :class:`torch.distributions.Distribution` and :class:`torch.nn.Module` for base distributions,
* :class:`torchflows.bijections.base.Bijection` for bijections,
* :class:`torchflows.flows.BaseFlow` for flow wrappers.

Check the following pages for existing subclasses and to learn to create new subclasses for your modeling and research needs:

.. toctree::
    :maxdepth: 1

    base_distributions
    bijections/index
    flow

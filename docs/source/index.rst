Welcome to Torchflows documentation!
===================================

Torchflows is a library for generative modeling and density estimation using normalizing flows.
It implements many normalizing flow architectures and their building blocks for:

* easy use of normalizing flows as trainable distributions;
* easy implementation of new normalizing flows.

Installing and usage
----------

Install Torchflows with pip:

.. code-block:: console

   pip install torchflows

Create a flow and train it as follows:
.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP

    x = torch.randn((1000, 25))  # generate synthetic 25-dimensional data
    flow = Flow(RealNVP((25,)))
    flow.fit(x, show_progress=True)

    x_new = flow.sample((150,))  # sample 150 new points from the flow

Contents
--------

.. toctree::

   usage
   api
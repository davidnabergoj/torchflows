Torchflows documentation
========================

Torchflows is a library for generative modeling and density estimation using normalizing flows.
It implements many normalizing flow architectures and their building blocks for:

* easy use of normalizing flows as trainable distributions;
* easy implementation of new normalizing flows.

Torchflows is structured according to the review paper `Normalizing Flows for Probabilistic Modeling and Inference <https://arxiv.org/abs/1912.02762>`_ by Papamakarios et al. (2021), which classifies flow architectures as autoregressive, residual, or continuous.
Visit the `Github page <https://github.com/davidnabergoj/torchflows>`_ to keep up to date and post any questions or issues `here <https://github.com/davidnabergoj/torchflows/issues>`_.

Installing
---------------
Torchflows can be installed easily using pip:

.. code-block:: console

   pip install torchflows

For other install options, see the :ref:`install <installing>` section.

Quick example
---------------

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    x_train = torch.randn(size=(1000, 3))  # 1000 three-dimensional data points

    flow = Flow(RealNVP(event_shape=(3,)))
    flow.fit(x_train, show_progress=True)

    x_new = flow.sample(50)  # (50, 3)
    log_prob = flow.log_prob(x_train)  # (1000,)

Table of contents
----------------------------

.. toctree::
    :maxdepth: 2

    guides/installing
    guides/tutorial
    architectures/index
    architectures/general_modeling
    architectures/image_modeling
    developer_reference/index

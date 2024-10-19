.. Torchflows documentation master file, created by
   sphinx-quickstart on Tue Aug 13 19:59:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torchflows documentation
========================

Torchflows is a library for generative modeling and density estimation using normalizing flows.
It implements many normalizing flow architectures and their building blocks for:

* easy use of normalizing flows as trainable distributions;
* easy implementation of new normalizing flows.

Torchflows is structured according to the review paper `Normalizing Flows for Probabilistic Modeling and Inference <(https://arxiv.org/abs/1912.02762)>`_ by Papamakarios et al. (2021), which classifies flow architectures as autoregressive, residual, or continuous.
Visit the `Github page <https://github.com/davidnabergoj/torchflows>`_ to keep up to date and post any questions or issues `here <https://github.com/davidnabergoj/torchflows/issues>`_.

Installing
---------------
Torchflows can be installed easily using pip:

.. code-block:: console

   pip install torchflows

For other install options, see the :ref:`install <installing>` section.

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


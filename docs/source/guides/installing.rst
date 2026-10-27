Installing
============================

.. _installing:

.. note::

    Torchflows supports Python versions 3.9 and upwards.

We provide several options to install Torchflows.

Install the latest stable version from PyPI:

.. code-block:: console

    pip install torchflows

Install the latest version from Github:

.. code-block:: console

    pip install git+https://github.com/davidnabergoj/torchflows.git

Install Torchflows for development (editable install with test dependencies):

.. code-block:: console

    git clone https://github.com/davidnabergoj/torchflows.git
    cd torchflows
    pip install -e .

Continuous normalizing flows require `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_, which is installed automatically as a dependency.

The installed version can be checked with:

.. code-block:: python

    import torchflows
    print(torchflows.__version__)

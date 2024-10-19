Modifying normalizing flow architectures
============================================

We sometimes wish to experiment with bijection parameters to improve NF performance on a given dataset.
We give a few examples on how to achieve this with Torchflows.

Passing hyperparameters to existing architecture constructors
-------------------------------------------------------------------
We can make basic modifications to an existing NF architecture by passing it certain keyword arguments.
The permitted keyword arguments depend on the architecture.
Suppose we are working with RealNVP, which is a composition of several affine coupling layers.
We wish our RealNVP instance to have 5 affine coupling layers.
Each affine coupling layer should use a feed-forward neural network conditioner with 5 layers, as well as 10 hidden neurons and the ReLU activation in each layer.

.. code-block:: python

    import torch.nn as nn
    from torchflows.flows import Flow
    from torchflows.architectures import RealNVP
    from torchflows.bijections.finite.autoregressive.conditioning.transforms import FeedForward

    event_shape = (10,)
    custom_hyperparameters = {
        'n_layers': 5,
        'conditioner_transform_class': FeedForward,
        'conditioner_kwargs': {
            'n_layers': 5,
            'n_hidden': 10,
            'nonlinearity': nn.ReLU
        }
    }
    bijection = RealNVP(event_shape, **custom_hyperparameters)
    flow = Flow(bijection)

`Autoregressive architectures <autoregressive_architecture_list>`_ can receive hyperparameters through the following keyword arguments:

* ``n_layers``: the number of affine coupling layer;
* ``conditioner_transform_class``: the conditioner type to use in each layer;
* ``conditioner_kwargs``: conditioner keyword arguments for each layer;
* ``transformer_kwargs``: transformer keyword arguments for each layer.

The specific keyword arguments depend on which conditioner and transformer we are using.
Check the list of implemented conditioner transforms and their constructors :doc:`here <../developer_reference/bijections/autoregressive/conditioner_transforms>`.
See which transformers are used in each architecture :ref:`here <autoregressive_architecture_list>`.

Coupling architectures can also receive:

* ``edge_list``: an edge list of conditional dimension interactions;
* ``coupling_kwargs``: keyword arguments for :func:`make_coupling` in each layer.

To see how other architectures use keyword arguments, consider checking the :doc:`list of architectures <../architectures/index>`

Composing existing bijections with custom hyperparameters
-------------------------------------------------------------
In the previous section, we learned how to modify a preset architecture by passing some hyperparameters.
In residual and autoregressive NFs, this approach will use the same hyperparameters for each layer of the NF.
For more customization, we can create individual layers and compose them into a custom architecture.
Suppose we wish to create a NF with five layers:

* two affine coupling layers,
* a rational quadratic spline coupling layer,
* an invertible residual network layer,
* an elementwise shift layer.

The above model can be coded as follows:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive.layers import AffineCoupling, RQSCoupling, ElementwiseShift
    from torchflows.bijections.finite.residual.iterative import InvertibleResNetBlock

    torch.manual_seed(0)
    event_shape = (10,)
    bijection = BijectiveComposition(
        event_shape,
        [
            AffineCoupling(event_shape),
            AffineCoupling(event_shape),
            RQSCoupling(event_shape),
            InvertibleResNetBlock(event_shape),
            ElementwiseShift(event_shape),
        ]
    )
    flow = Flow(bijection)

    x_new = flow.sample((10,))
    log_prob = flow.log_prob(x_new)

We can also customize each layer with custom hyperparameters, for example:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive.layers import AffineCoupling, RQSCoupling, ElementwiseShift
    from torchflows.bijections.finite.residual.iterative import InvertibleResNetBlock

    torch.manual_seed(0)
    event_shape = (10,)
    bijection = BijectiveComposition(
        event_shape,
        [
            AffineCoupling(event_shape, conditioner_kwargs={'n_hidden': 5, 'n_layers': 10}),
            AffineCoupling(event_shape),
            RQSCoupling(event_shape, conditioner_kwargs={'n_layers': 1}),
            InvertibleResNetBlock(event_shape, hidden_size=4, n_hidden_layers=3),
            ElementwiseShift(event_shape),
        ]
    )
    flow = Flow(bijection)

    x_new = flow.sample((10,))
    log_prob = flow.log_prob(x_new)

.. note::

    Due to the large number of bijections in the library, argument names are not always consistent across bijections.
    Check bijection constructors to make sure you are using correct argument names.
    We are working to improve this in a future release.

Composing NF architectures
----------------------------------------

Since each NF transformation is a bijection, we can compose them as any other.
We give an example below, where we compose RealNVP, coupling RQ-NSF, FFJORD, and ResFlow:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive.architectures import RealNVP, CouplingRQNSF
    from torchflows.bijections.finite.residual.architectures import ResFlow
    from torchflows.bijections.continuous.ffjord import FFJORD

    torch.manual_seed(0)
    event_shape = (10,)
    bijection = BijectiveComposition(
        event_shape,
        [
            RealNVP(event_shape),
            CouplingRQNSF(event_shape),
            FFJORD(event_shape),
            ResFlow(event_shape)
        ]
    )
    flow = Flow(bijection)

    x_new = flow.sample((10,))
    log_prob = flow.log_prob(x_new)

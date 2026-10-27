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
    from torchflows.bijections.finite.autoregressive import RealNVP
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

:ref:`Autoregressive architectures <autoregressive_architecture_list>` can receive hyperparameters through the following keyword arguments:

* ``n_layers``: the number of autoregressive layers;
* ``conditioner_transform_class``: the conditioner type to use in each layer;
* ``conditioner_kwargs``: conditioner keyword arguments for each layer;
* ``transformer_kwargs``: transformer keyword arguments for each layer;
* ``l2_regularization``: if True, add an L2 penalty on conditioner parameters to the training loss (enabled by default for coupling and masked autoregressive layers);
* ``l2_coef``: the L2 penalty coefficient.

The specific keyword arguments depend on which conditioner and transformer we are using.
Check the list of implemented conditioner transforms and their constructors :doc:`here <../developer_reference/bijections/autoregressive/conditioner_transforms>`.
See which transformers are used in each architecture :ref:`here <autoregressive_architecture_list>`.

Coupling architectures can also receive:

* ``edge_list``: an edge list of conditional dimension interactions;
* ``coupling_kwargs``: keyword arguments for :func:`~torchflows.bijections.finite.autoregressive.conditioning.coupling_masks.make_coupling` in each layer.

:ref:`Residual architectures <residual_architecture_api>` receive the number of layers through ``n_layers`` and per-layer keyword arguments through ``layer_kwargs``:

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.bijections.finite.residual import ResFlow

    event_shape = (10,)
    bijection = ResFlow(event_shape, n_layers=4, layer_kwargs={'hidden_size': 20})
    flow = Flow(bijection)

To see how other architectures use keyword arguments, consider checking the :doc:`list of architectures <../architectures/index>`.

Composing existing bijections with custom hyperparameters
-------------------------------------------------------------
In the previous section, we learned how to modify a preset architecture by passing some hyperparameters.
In residual and autoregressive NFs, this approach will use the same hyperparameters for each layer of the NF.
For more customization, we can create individual layers and compose them into a custom architecture with :class:`~torchflows.bijections.base.BijectiveComposition`.
The composition infers its event shape from the first layer.
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
    bijection = BijectiveComposition([
        AffineCoupling(event_shape),
        AffineCoupling(event_shape),
        RQSCoupling(event_shape),
        InvertibleResNetBlock(event_shape),
        ElementwiseShift(event_shape),
    ])
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
    bijection = BijectiveComposition([
        AffineCoupling(event_shape, conditioner_kwargs={'n_hidden': 5, 'n_layers': 10}),
        AffineCoupling(event_shape),
        RQSCoupling(event_shape, conditioner_kwargs={'n_layers': 1}, transformer_kwargs={'n_bins': 16}),
        InvertibleResNetBlock(event_shape, hidden_size=4, n_hidden_layers=3),
        ElementwiseShift(event_shape),
    ])
    flow = Flow(bijection)

    x_new = flow.sample((10,))
    log_prob = flow.log_prob(x_new)

Permutations between coupling layers ensure that every dimension gets transformed.
Torchflows provides permutations and other linear bijections in :mod:`torchflows.bijections.finite.matrix`:

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive.layers import AffineCoupling
    from torchflows.bijections.finite.matrix import ReversePermutationMatrix, LUMatrix

    event_shape = (10,)
    bijection = BijectiveComposition([
        AffineCoupling(event_shape),
        ReversePermutationMatrix(event_shape),
        AffineCoupling(event_shape),
        LUMatrix(event_shape),  # trainable invertible linear map
        AffineCoupling(event_shape),
    ])
    flow = Flow(bijection)

.. note::

    Due to the large number of bijections in the library, argument names are not always consistent across bijections.
    Check bijection constructors to make sure you are using correct argument names.
    We are working to improve this in a future release.

Composing NF architectures
----------------------------------------

Since each NF transformation is a bijection, we can compose them as any other.
We give an example below, where we compose RealNVP, coupling RQ-NSF, and ResFlow:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.base import BijectiveComposition
    from torchflows.bijections.finite.autoregressive import RealNVP, CouplingRQNSF
    from torchflows.bijections.finite.residual import ResFlow

    torch.manual_seed(0)
    event_shape = (10,)
    bijection = BijectiveComposition([
        RealNVP(event_shape),
        CouplingRQNSF(event_shape),
        ResFlow(event_shape)
    ])
    flow = Flow(bijection)

    x_new = flow.sample((10,))
    log_prob = flow.log_prob(x_new)

.. note::

    Continuous bijections (FFJORD, RNODE, OT-Flow, DDNF) should be wrapped with :class:`~torchflows.flows.ContinuousFlow` instead of being composed with finite bijections.
    See :doc:`continuous_flows`.

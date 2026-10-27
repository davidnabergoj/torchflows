Continuous normalizing flows
================================

Continuous normalizing flows (CNFs) define the bijection as the solution of an ordinary differential equation (ODE).
Torchflows integrates the ODE with `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_.
Continuous bijections are found in :mod:`torchflows.bijections.continuous` and are wrapped with :class:`~torchflows.flows.ContinuousFlow` instead of :class:`~torchflows.flows.Flow`.
``ContinuousFlow`` passes auxiliary ODE outputs (e.g., Jacobian norms or transport costs) to the regularization term during training.

Fitting a CNF
--------------------

.. code-block:: python

    import torch
    from torchflows.flows import ContinuousFlow
    from torchflows.bijections.continuous import FFJORD

    torch.manual_seed(0)
    x_train = torch.randn(size=(1000, 3))

    flow = ContinuousFlow(FFJORD(event_shape=(3,)))
    flow.fit(x_train, n_epochs=50, show_progress=True)

    x_new = flow.sample(100)
    print(x_new.shape)  # (100, 3)

    # ContinuousFlow.log_prob returns a tuple: the log probability, followed by optional auxiliary tensors
    log_prob = flow.log_prob(x_train)[0]
    print(log_prob.shape)  # (1000,)

The available presets are:

* :class:`~torchflows.bijections.continuous.ffjord.FFJORD`: Hutchinson trace estimation with Jacobian regularization;
* :class:`~torchflows.bijections.continuous.rnode.RNODE`: regularized neural ODE;
* :class:`~torchflows.bijections.continuous.ddnf.DDNF`: deep diffeomorphic normalizing flow (Euler steps);
* :class:`~torchflows.bijections.continuous.otflow.OTFlowBijection`: OT-Flow with exact trace and optimal transport regularization;
* convolutional variants for images: ``ConvolutionalFFJORD``, ``ConvolutionalRNODE``, ``ConvolutionalDDNF``.

OT-Flow
--------------------

OT-Flow regularizes the transport cost and the Hamilton-Jacobi-Bellman (HJB) residual.
Regularization coefficients are set through ``time_derivative_kwargs``:

.. code-block:: python

    import torch
    from torchflows.flows import ContinuousFlow
    from torchflows.bijections.continuous import OTFlowBijection

    torch.manual_seed(0)
    x_train = torch.randn(size=(1000, 2))

    bijection = OTFlowBijection(
        event_shape=(2,),
        time_derivative_kwargs={
            'reg_transport_coef': 0.05,
            'reg_hjb_coef': 0.01,
        }
    )
    flow = ContinuousFlow(bijection)
    flow.fit(x_train, n_epochs=50, show_progress=True)

Solver settings
--------------------

All continuous bijections accept ODE solver arguments:

* ``solver``: any `torchdiffeq solver <https://github.com/rtqichen/torchdiffeq#list-of-ode-solvers>`_, e.g., ``'rk4'`` (default for most presets), ``'dopri5'``, ``'euler'``;
* ``atol``, ``rtol``: absolute and relative tolerances for adaptive solvers;
* ``end_time``: integration end time (default 1).

The neural network that parameterizes the time derivative is configured with ``nn_kwargs`` and Hutchinson trace estimation with ``time_derivative_kwargs``:

.. code-block:: python

    from torchflows.flows import ContinuousFlow
    from torchflows.bijections.continuous import RNODE

    bijection = RNODE(
        event_shape=(5,),
        solver='dopri5',
        atol=1e-5,
        rtol=1e-5,
        nn_kwargs={'hidden_size': 32, 'n_hidden_layers': 2},
        time_derivative_kwargs={'n_noise_samples': 2, 'reg_jac_coef': 0.05}
    )
    flow = ContinuousFlow(bijection)

Creating a custom CNF
--------------------------

A custom CNF combines a time derivative module with :class:`~torchflows.bijections.continuous.base.ContinuousBijection`.
Below, we use a feed-forward time derivative network and estimate the divergence with the Hutchinson estimator.
To add regularization, subclass ``ContinuousBijection`` and implement ``regularization`` (see the :doc:`continuous bijection reference <../developer_reference/bijections/continuous/bijections>`).

.. code-block:: python

    import torch
    from torchflows.flows import ContinuousFlow
    from torchflows.bijections.continuous.base import (
        ContinuousBijection,
        HutchinsonTimeDerivative,
        create_dnn_forward_model
    )

    event_shape = (3,)
    time_derivative = HutchinsonTimeDerivative(
        forward_model=create_dnn_forward_model(event_shape, hidden_size=16, n_hidden_layers=1),
        n_noise_samples=1
    )
    bijection = ContinuousBijection(event_shape, f=time_derivative, solver='rk4')
    flow = ContinuousFlow(bijection)

    torch.manual_seed(0)
    flow.fit(torch.randn(size=(500, *event_shape)), n_epochs=20)

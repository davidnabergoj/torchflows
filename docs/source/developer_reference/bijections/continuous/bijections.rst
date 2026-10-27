Continuous bijections
===========================

Continuous bijections integrate a time derivative :math:`dx/dt = f(t, x)` from time 0 to ``end_time``.
The log determinant is the integral of the divergence of :math:`f`.
Wrap continuous bijections with :class:`~torchflows.flows.ContinuousFlow`.

.. autoclass:: torchflows.bijections.continuous.base.ContinuousBijection
    :members: __init__, forward, inverse

Time derivatives
-------------------

A time derivative object computes :math:`dx/dt` and the divergence in each ODE step.
Subclasses implement ``prepare_initial_state`` and ``step``.

.. autoclass:: torchflows.bijections.continuous.time_derivative.TimeDerivative
    :members: prepare_initial_state, step

.. autoclass:: torchflows.bijections.continuous.base.HutchinsonTimeDerivative
    :members: __init__

.. autoclass:: torchflows.bijections.continuous.otflow.OTFlowTimeDerivative
    :members: __init__

Time derivative modules
--------------------------

Time derivative modules are neural network layers that receive time ``t`` and space ``x``.
They are combined with :class:`~torchflows.bijections.continuous.time_derivative.TimeDerivativeSequential`.

.. autoclass:: torchflows.bijections.continuous.time_derivative.TimeDerivativeModule
    :members: forward

.. autoclass:: torchflows.bijections.continuous.time_derivative.TimeDerivativeSequential

.. autofunction:: torchflows.bijections.continuous.base.create_dnn_forward_model

Example: a custom CNF with a hand-built time derivative network and Jacobian regularization.
When the time derivative returns auxiliary tensors (here, the squared Jacobian norm), the bijection must implement ``regularization`` to turn them into a scalar penalty.

.. code-block:: python

    import torch
    from torchflows.flows import ContinuousFlow
    from torchflows.bijections.continuous.base import ContinuousBijection, HutchinsonTimeDerivative, TimeTanh
    from torchflows.bijections.continuous.time_derivative import TimeDerivativeSequential, ConcatLinear

    class MyCNF(ContinuousBijection):
        def __init__(self, event_shape, hidden_size: int = 32, **kwargs):
            hidden_shape = (hidden_size,)
            network = TimeDerivativeSequential([
                ConcatLinear(event_shape, hidden_shape),
                TimeTanh(),
                ConcatLinear(hidden_shape, event_shape),
            ])
            super().__init__(
                event_shape,
                f=HutchinsonTimeDerivative(forward_model=network, reg_jac=True, reg_jac_coef=0.01),
                **kwargs
            )

        def regularization(self, sq_jac_norm: torch.Tensor = None):
            if sq_jac_norm is None:
                return torch.tensor(0.0)
            return self.f.reg_jac_coef * sq_jac_norm.mean()

    event_shape = (2,)
    flow = ContinuousFlow(MyCNF(event_shape, solver='rk4'))
    flow.fit(torch.randn(size=(500, *event_shape)), n_epochs=20)

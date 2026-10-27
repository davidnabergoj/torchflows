Residual bijections
===========================

Residual bijections have the form :math:`z = x + g(x)`.
Torchflows distinguishes two kinds:

* iterative residual bijections (i-ResNet, ResFlow, proximal ResFlow), where :math:`g` is contractive, the inverse is computed with fixed-point iteration, and the log determinant is estimated stochastically;
* classic residual bijections (planar, radial, Sylvester), where the log determinant is computed exactly with the matrix determinant lemma, but only one direction is implemented.

Residual architectures compose residual layers with elementwise affine layers:

.. autoclass:: torchflows.bijections.finite.residual.base.ResidualArchitecture
    :members: __init__

Iterative residual bijections
------------------------------------

.. autoclass:: torchflows.bijections.finite.residual.base.IterativeResidualBijection
    :members: forward, inverse, log_det

.. autoclass:: torchflows.bijections.finite.residual.iterative.InvertibleResNetBlock
.. autoclass:: torchflows.bijections.finite.residual.iterative.ConvolutionalInvertibleResNetBlock
.. autoclass:: torchflows.bijections.finite.residual.iterative.ResFlowBlock
.. autoclass:: torchflows.bijections.finite.residual.iterative.ConvolutionalResFlowBlock
.. autoclass:: torchflows.bijections.finite.residual.proximal.ProximalResFlowBlock

Log determinant estimators
------------------------------------

Iterative residual bijections estimate :math:`\log|\det(I + J_g(x))|` with a power series (i-ResNet) or a Russian roulette estimator (ResFlow).

.. autofunction:: torchflows.bijections.finite.residual.log_abs_det_estimators.log_det_power_series
.. autofunction:: torchflows.bijections.finite.residual.log_abs_det_estimators.log_det_roulette

A new iterative residual block defines a contractive function ``g`` and a ``log_det`` method:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.residual.base import IterativeResidualBijection
    from torchflows.bijections.finite.residual.iterative import SpectralNeuralNetwork
    from torchflows.bijections.finite.residual.log_abs_det_estimators import log_det_power_series

    class MyResidualBlock(IterativeResidualBijection):
        def __init__(self, event_shape, **kwargs):
            super().__init__(event_shape)
            self.g = SpectralNeuralNetwork(event_shape, hidden_size=16, n_hidden_layers=1)

        def log_det(self, x, **kwargs):
            return log_det_power_series(self.event_shape, self.g, x, n_iterations=5, **kwargs)[1]

    flow = Flow(MyResidualBlock(event_shape=(4,)))
    flow.fit(torch.randn(size=(200, 4)), n_epochs=10)

Classic residual bijections
------------------------------------

.. autoclass:: torchflows.bijections.finite.residual.base.ClassicResidualBijection
    :members: __init__

.. autoclass:: torchflows.bijections.finite.residual.planar.Planar
.. autoclass:: torchflows.bijections.finite.residual.radial.Radial
.. autoclass:: torchflows.bijections.finite.residual.sylvester.BaseSylvester
.. autoclass:: torchflows.bijections.finite.residual.sylvester.HouseholderSylvester
.. autoclass:: torchflows.bijections.finite.residual.sylvester.IdentitySylvester
.. autoclass:: torchflows.bijections.finite.residual.sylvester.PermutationSylvester

Classic residual bijections only implement the inverse pass, so they are used for sampling and variational inference:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.residual import PlanarBijection

    flow = Flow(PlanarBijection(event_shape=(2,), n_layers=8))
    x, log_prob = flow.sample(100, return_log_prob=True)
    print(x.shape, log_prob.shape)  # (100, 2) (100,)

List of transformers
================================

Torchflows supports several transformers to be used in autoregressive and multiscale normalizing flows.

Linear transformers
--------------------
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.linear.affine.Affine
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.linear.affine.InverseAffine
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.linear.affine.Shift
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.linear.convolution.Invertible1x1ConvolutionTransformer
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.linear.matrix.LUTransformer

Spline transformers
--------------------------------
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.spline.linear.Linear
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.spline.linear_rational.LinearRational
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.spline.rational_quadratic.RationalQuadratic

Combination transformers
---------------------------------------

.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid.Sigmoid
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid.DenseSigmoid
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid.DeepSigmoid
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid.DeepDenseSigmoid

Integration transformers
---------------------------------
.. autoclass:: torchflows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network.UnconstrainedMonotonicNeuralNetwork

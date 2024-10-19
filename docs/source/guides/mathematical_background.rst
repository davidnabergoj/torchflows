Mathematical background
=============================


What is a normalizing flow
--------------------------------------------

A normalizing flow (NF) is a flexible trainable distribution.
It is defined as a bijective transformation of a simple distribution, such as a standard Gaussian.
The bijection is typically an invertible neural network.
Training a NF using a dataset means optimizing the bijection's parameters to make the dataset likely under the NF.
We can use a NF to compute the probability of a data point or to independently sample data from the process that
generated our dataset.

The density of a NF :math:`q(x)` with the bijection :math:`f(z) = x` and base distribution :math:`p(z)` is defined as:

.. math::
    \log q(x) = \log p(f^{-1}(x)) + \log\left|\det J_{f^{-1}}(x)\right|.

Sampling from a NF means sampling from the simple distribution and transforming the sample using the bijection.

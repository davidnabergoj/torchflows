Flow objects
===============================
The `Flow` object contains a base distribution and a bijection.

.. _flow:

.. autoclass:: torchflows.flows.BaseFlow
    :members: regularization, fit, variational_fit

.. autoclass:: torchflows.flows.Flow
    :members: __init__, forward_with_log_prob, log_prob, sample

.. autoclass:: torchflows.flows.FlowMixture
    :members: __init__, log_prob, sample
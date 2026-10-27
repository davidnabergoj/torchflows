Flow wrappers
===============================
The ``Flow`` object contains a base distribution and a bijection.
Use :class:`~torchflows.flows.Flow` for finite bijections and :class:`~torchflows.flows.ContinuousFlow` for continuous bijections.

.. _flow:

.. autoclass:: torchflows.flows.BaseFlow
    :members: regularization, fit, variational_fit, fit_kl_p_to_q

.. autoclass:: torchflows.flows.Flow
    :members: __init__, forward_with_log_prob, log_prob, sample

.. autoclass:: torchflows.flows.ContinuousFlow
    :members: __init__, forward_with_log_prob, log_prob

.. autoclass:: torchflows.flows.FlowMixture
    :members: __init__, log_prob, sample

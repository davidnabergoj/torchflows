Variational inference
========================

Instead of fitting a flow to data, we can fit it to an unnormalized target density.
This is known as (stochastic) variational inference and is useful when we can evaluate a density but cannot sample from it, e.g., a Bayesian posterior.
The method :meth:`~torchflows.flows.BaseFlow.variational_fit` minimizes the reverse KL divergence between the flow and the target.
The target is a function that maps a tensor with shape ``(*batch_shape, *event_shape)`` to log densities with shape ``batch_shape``.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)

    def target_log_prob(x: torch.Tensor) -> torch.Tensor:
        # Banana-shaped density in 2D
        x0, x1 = x[..., 0], x[..., 1]
        return -0.5 * x0 ** 2 - 0.5 * (x1 - x0 ** 2) ** 2

    flow = Flow(RealNVP(event_shape=(2,)))
    flow.variational_fit(
        target_log_prob,
        n_epochs=500,
        lr=0.01,
        n_samples=64,  # samples per gradient step
        show_progress=True
    )

    x_approx = flow.sample(1000, no_grad=True)
    print(x_approx.shape)  # (1000, 2)

Variational inference requires a fast inverse pass (sampling).
Inverse autoregressive architectures such as ``IAF`` or ``InverseAutoregressiveRQNSF``, as well as ``PlanarBijection``, ``RadialBijection`` and ``SylvesterBijection``, are well suited for it.

.. code-block:: python

    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import IAF
    from torchflows.bijections.finite.residual import PlanarBijection

    flow_iaf = Flow(IAF(event_shape=(2,)))
    flow_iaf.variational_fit(target_log_prob, n_epochs=200)

    flow_planar = Flow(PlanarBijection(event_shape=(2,)))
    flow_planar.variational_fit(target_log_prob, n_epochs=200)

.. note::

    Planar, radial and Sylvester bijections only implement the inverse pass.
    They support sampling with log probabilities (``flow.sample(n, return_log_prob=True)``), but not ``flow.log_prob(x)`` on arbitrary data.

Fitting with target samples
--------------------------------

When we have samples from the target and can evaluate the target density, we can minimize the forward KL divergence with :meth:`~torchflows.flows.BaseFlow.fit_kl_p_to_q`:

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import RealNVP

    torch.manual_seed(0)
    x = torch.randn(size=(1200, 2))

    def neg_log_prob_target(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x ** 2, dim=-1)

    flow = Flow(RealNVP(event_shape=(2,)))
    flow.fit_kl_p_to_q(x[:1000], x[1000:], neg_log_prob_target, n_epochs=100)

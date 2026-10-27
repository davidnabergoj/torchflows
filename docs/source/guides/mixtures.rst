Mixtures of normalizing flows
==================================

:class:`~torchflows.flows.FlowMixture` combines several flows into a statistical mixture.
Each component can use a different architecture.
Mixtures can help model multimodal data.

.. code-block:: python

    import torch
    from torchflows.flows import Flow, FlowMixture
    from torchflows.bijections.finite.autoregressive import RealNVP, MAF

    torch.manual_seed(0)
    x_train = torch.cat([
        torch.randn(size=(500, 2)) - 3,
        torch.randn(size=(500, 2)) + 3,
    ])

    mixture = FlowMixture([
        Flow(RealNVP(event_shape=(2,))),
        Flow(MAF(event_shape=(2,))),
    ])
    mixture.fit(x_train, n_epochs=100, show_progress=True)

    x_new = mixture.sample(50)
    print(x_new.shape)  # (50, 2)

    log_prob = mixture.log_prob(x_train)
    print(log_prob.shape)  # (1000,)

By default, all components have equal weights.
Initial weights can be set with ``weights``, and ``trainable_weights=True`` learns them during training.

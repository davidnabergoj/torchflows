Conditional normalizing flows
==================================

A conditional flow models the density of data ``x`` given a context tensor ``c``.
To create one, pass ``context_shape`` to the bijection constructor and provide context tensors when fitting, computing log probabilities, and sampling.
Context support is available in autoregressive architectures.

.. code-block:: python

    import torch
    from torchflows.flows import Flow
    from torchflows.bijections.finite.autoregressive import MAF

    torch.manual_seed(0)
    n_data = 1000
    context = torch.randn(size=(n_data, 2))
    x = context.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(size=(n_data, 3))  # x depends on context

    flow = Flow(MAF(event_shape=(3,), context_shape=(2,)))
    flow.fit(x, context_train=context, n_epochs=100, show_progress=True)

    log_prob = flow.log_prob(x, context=context)
    print(log_prob.shape)  # (1000,)

    # Draw one sample for each of the first 10 context vectors
    x_new = flow.sample(10, context=context[:10])
    print(x_new.shape)  # (10, 3)

Validation data is passed with matching context tensors:

.. code-block:: python

    flow.fit(
        x[:800], context_train=context[:800],
        x_val=x[800:], context_val=context[800:],
        n_epochs=100
    )

.. note::

    When sampling, the number of samples must match the number of context vectors.
    To draw many samples for a single context vector, repeat it:
    ``flow.sample(100, context=c[None].repeat(100, 1))``.

Numerical stability
=============================

We may require a bijection to be very numerically precise when transforming data between original and latent spaces.
Given data `x`, bijection `b`, and tolerance `epsilon`, we may want:

.. code-block:: python

    z, log_det_forward = b.forward(x)
    x_reconstructed, log_det_inverse = b.inverse(z)

    assert torch.all(torch.abs(x_reconstructed - x) < epsilon)
    assert torch.all(torch.abs(log_det_forward + log_det_inverse)) < epsilon)


All architecture presets in Torchflows (with a defined forward and inverse pass) are tested to reconstruct inputs and log determinants.
We test reconstruction with inputs taken from a standard Gaussian distribution.
The specified tolerance is either 0.01 or 0.001, though many architectures achieve a lower reconstruction error.


Reducing reconstruction error
------------------------------------------

We may need an even smaller reconstruction error.
We can start by ensuring the input data is standardized:

.. code-block:: python

    import torch
    from torchflows.architectures import RealNVP

    torch.manual_seed(0)

    batch_shape = (5,)
    event_shape = (10,)
    x = (torch.randn(size=(*batch_shape, *event_shape)) * 12 + 35) ** 0.5
    x_standardized = (x - x.mean()) / x.std()

    real_nvp = RealNVP(event_shape)

    def print_reconstruction_errors(bijection, inputs):
        z, log_det_forward = bijection.forward(inputs)
        inputs_reconstructed, log_det_inverse = bijection.inverse(z)

        print(f'Data reconstruction error: {torch.max(torch.abs(inputs - inputs_reconstructed)):.8f}')
        print(f'Log determinant error: {torch.max(torch.abs(log_det_forward + log_det_inverse)):.8f}')

    # Test with non-standardized inputs
    print_reconstruction_errors(real_nvp, x)
    print('-------------------------------------------------------')
    print_reconstruction_errors(real_nvp, x_standardized)

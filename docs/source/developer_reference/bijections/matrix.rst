Invertible linear bijections
================================

The module :mod:`torchflows.bijections.finite.matrix` contains linear bijections :math:`z = Wx` with structured invertible matrices :math:`W`.
They are used as permutations or trainable mixing layers between coupling layers.
All classes receive the event shape as the first argument.

.. code-block:: python

    import torch
    from torchflows.bijections.finite.matrix import LUMatrix

    bijection = LUMatrix(event_shape=(5,))
    x = torch.randn(size=(10, 5))
    z, log_det = bijection.forward(x)
    print(z.shape, log_det.shape)  # (10, 5) (10,)

Base class
--------------

.. autoclass:: torchflows.bijections.finite.matrix.base.InvertibleMatrix
    :members: __init__

Permutations
--------------

.. autoclass:: torchflows.bijections.finite.matrix.permutation.ReversePermutationMatrix
.. autoclass:: torchflows.bijections.finite.matrix.permutation.RandomPermutationMatrix

Trainable matrices
---------------------

.. autoclass:: torchflows.bijections.finite.matrix.identity.IdentityMatrix
.. autoclass:: torchflows.bijections.finite.matrix.decomposition.LUMatrix
.. autoclass:: torchflows.bijections.finite.matrix.decomposition.QRMatrix
.. autoclass:: torchflows.bijections.finite.matrix.orthogonal.HouseholderProductMatrix
.. autoclass:: torchflows.bijections.finite.matrix.triangular.LowerTriangularInvertibleMatrix
.. autoclass:: torchflows.bijections.finite.matrix.triangular.UpperTriangularInvertibleMatrix

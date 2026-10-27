# Creating and modifying bijection architectures

We give an example on how to modify a bijection's architecture.
We use the Masked Autoregressive Flow (MAF) as an example.
We can manually set the number of invertible layers as follows:

```python
from torchflows.bijections.finite.autoregressive import MAF

event_shape = (10,)
bijection = MAF(event_shape=event_shape, n_layers=5)
```

For specific changes, we can create individual invertible layers and combine them into a bijection.
MAF uses affine masked autoregressive layers with permutations in between.
We can import these layers and set their parameters as desired.
For example, to change the number of layers in the MAF conditioner (MADE) and its hidden layer sizes, we proceed as follows:

```python
from torchflows.flows import Flow
from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers import AffineForwardMaskedAutoregressive
from torchflows.bijections.finite.matrix import ReversePermutationMatrix

event_shape = (10,)
bijection = BijectiveComposition([
    AffineForwardMaskedAutoregressive(event_shape, conditioner_kwargs={'n_layers': 4, 'n_hidden': 20}),
    ReversePermutationMatrix(event_shape),
    AffineForwardMaskedAutoregressive(event_shape, conditioner_kwargs={'n_layers': 3, 'n_hidden': 7}),
    ReversePermutationMatrix(event_shape),
    AffineForwardMaskedAutoregressive(event_shape, conditioner_kwargs={'n_layers': 5, 'n_hidden': 13}),
])
flow = Flow(bijection)
```

# Creating and modifying bijection architectures

We give an example on how to modify a bijection's architecture.
We use the Masked Autoregressive Flow (MAF) as an example.
We can manually set the number of invertible layers as follows:
```python
from normalizing_flows.bijections import MAF

event_shape = (10,)
flow = MAF(event_shape=event_shape, n_layers=5)
```

For specific changes, we can create individual invertible layers and combine them into a bijection.
MAF uses affine masked autoregressive layers with permutations in between.
We can import these layers set their parameters as desired.
For example, to change the number of layers in the MAF conditioner and its hidden layer sizes, we proceed as follows: 
```python
from normalizing_flows.bijections import BijectiveComposition
from normalizing_flows.bijections.finite.autoregressive.layers import AffineForwardMaskedAutoregressive
from normalizing_flows.bijections.finite.linear import ReversePermutation

event_shape = (10,)
flow = BijectiveComposition(
    event_shape=event_shape,
    layers=[
        AffineForwardMaskedAutoregressive(event_shape=event_shape, n_layers=4, n_hidden=20),
        ReversePermutation(event_shape=event_shape),
        AffineForwardMaskedAutoregressive(event_shape=event_shape, n_layers=3, n_hidden=7),
        ReversePermutation(event_shape=event_shape),
        AffineForwardMaskedAutoregressive(event_shape=event_shape, n_layers=5, n_hidden=13)
    ]
)
```
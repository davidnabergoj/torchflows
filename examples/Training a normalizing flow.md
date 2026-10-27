# Training normalizing flow models on a dataset

We demonstrate how to train a normalizing flow on a dataset.
We use Real NVP as an example, but you can replace it with any other bijection from `torchflows.bijections`.
The code is as follows:

```python
import torch
from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive import RealNVP

torch.manual_seed(0)

# We support arbitrary event shapes
event_shape = (2, 3)
n_data = 500
x_train = torch.randn(size=(n_data, *event_shape))

bijection = RealNVP(event_shape=event_shape)
flow = Flow(bijection)

flow.fit(x_train, show_progress=True)
```

To modify the learning rate, simply use the `lr` keyword argument in `flow.fit(...)`:

```python
flow.fit(x_train, show_progress=True, lr=0.001)
```

To use validation data with early stopping, pass `x_val` and `early_stopping=True`:

```python
x_val = torch.randn(size=(100, *event_shape))
flow.fit(x_train, x_val=x_val, early_stopping=True, show_progress=True)
```

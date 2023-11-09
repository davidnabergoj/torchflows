# Training normalizing flow models on a dataset

We demonstrate how to train a normalizing flow on a dataset.
We use Real NVP as an example, but you can replace it with any other bijection from `normalizing_flows.bijections`.
The code is as follows:

```python
import torch
from normalizing_flows import Flow
from normalizing_flows.architectures import RealNVP

torch.manual_seed(0)

# We support arbitrary event and batch shapes
event_shape = (2, 3)
batch_shape = (5, 7)
x_train = torch.randn(size=(*batch_shape, *event_shape))

bijection = RealNVP(event_shape=event_shape)
flow = Flow(bijection)

flow.fit(x_train, show_progress=True)
```

To modify the learning rate, simply use the `lr` keyword argument in `flow.fit(...)`:

```python
flow.fit(x_train, show_progress=True, lr=0.001)
```
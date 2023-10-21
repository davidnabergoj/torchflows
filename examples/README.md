# Examples

We provide minimal working examples on how to perform various common tasks with normalizing flows.
We use Real NVP as an example, but you can replace it with any other bijection from `normalizing_flows.bijections`.

## Training a normalizing flow on a fixed dataset
```python
import torch
from normalizing_flows import Flow
from normalizing_flows.bijections import RealNVP

torch.manual_seed(0)

# We support arbitrary event and batch shapes
event_shape = (2, 3)
batch_shape = (5, 7)
x_train = torch.randn(size=(*batch_shape, *event_shape))

bijection = RealNVP(event_shape=event_shape)
flow = Flow(bijection)

flow.fit(x_train, show_progress=True)
```

## Computing the log determinant of the Jacobian transformation given a Flow
```python
import torch
from normalizing_flows import Flow
from normalizing_flows.bijections import RealNVP

torch.manual_seed(0)

batch_shape = (5, 7)
event_shape = (2, 3)
x = torch.randn(size=(*batch_shape, *event_shape))
z = torch.randn(size=(*batch_shape, *event_shape))

bijection = RealNVP(event_shape=event_shape)
flow = Flow(bijection)

_, log_det_forward = flow.bijection.forward(x)
# log_det_forward.shape == batch_shape

_, log_det_inverse = flow.bijection.inverse(z)
# log_det_inverse.shape == batch_shape
```
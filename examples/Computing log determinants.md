# Computing the log determinant of the Jacobian

We show how to compute and retrieve the log determinant of the Jacobian of a bijective transformation. 
We use Real NVP as an example, but you can replace it with any other bijection from `torchflows.bijections`.
The code is as follows:

```python
import torch
from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive import RealNVP

torch.manual_seed(0)

batch_shape = (5, 7)
event_shape = (2, 3)
x = torch.randn(size=(*batch_shape, *event_shape))
z = torch.randn(size=(*batch_shape, *event_shape))

bijection = RealNVP(event_shape=event_shape)
flow = Flow(bijection)

_, log_det_forward = flow.bijection.forward(x)
_, log_det_inverse = flow.bijection.inverse(z)

print(log_det_forward.shape)  # (5, 7)
print(log_det_inverse.shape)  # (5, 7)
```

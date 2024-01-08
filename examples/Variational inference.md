We show how to fit normalizing flows using stochastic variational inference (SVI). Whereas traditional maximum
likelihood estimation requires a fixed dataset of samples, SVI lets us optimize NF parameters with the unnormalized
target log density function.

As an example, we define the unnormalized log density of a diagonal Gaussian. We assume this target has 10 dimensions
with mean 5 and variance 9 in each dimension:

```python
import torch

torch.manual_seed(0)

event_shape = (10,)
true_mean = torch.full(size=event_shape, fill_value=5.0)
true_variance = torch.full(size=event_shape, fill_value=9.0)


def target_log_prob(x: torch.Tensor):
    return torch.sum(-((x - true_mean) ** 2 / (2 * true_variance)), dim=1)
```

We define the flow and run the variational fit:

```python
from normalizing_flows import Flow
from normalizing_flows.bijections import RealNVP

torch.manual_seed(0)
flow = Flow(RealNVP(event_shape=event_shape))
flow.variational_fit(target_log_prob, show_progress=True)
```

We plot samples from the trained flow. We also print estimated marginal means and variances. We see that the estimates are roughly accurate.
```python
import matplotlib.pyplot as plt

torch.manual_seed(0)
x_flow = flow.sample(10000).detach()

plt.figure()
plt.scatter(x_flow[:, 0], x_flow[:, 1])
plt.show()

print(f'{torch.mean(x_flow, dim=0) = }')
print(f'{torch.var(x_flow, dim=0) = }')
```
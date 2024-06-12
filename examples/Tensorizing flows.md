```python
import torch
from normalizing_flows import Flow
from normalizing_flows.architectures import RealNVP
from normalizing_flows.base_distributions.tensor_train import UnconstrainedTensorTrain

loc = 1.0
scale = 2.0


def target_log_probability(x):
    return -torch.sum(0.5 * ((x - loc) ** 2) / scale ** 2, dim=1)


torch.manual_seed(0)

event_shape = (2,)

bijection = RealNVP(event_shape=event_shape)
base = UnconstrainedTensorTrain(event_shape, target_log_probability)
flow = Flow(bijection, base_distribution=base)

flow.variational_fit(target_log_probability, show_progress=True)

x_flow = flow.sample(10000).detach()
print(x_flow.mean(dim=0))
print(x_flow.std(dim=0))
```
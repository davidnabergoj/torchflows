# Torchflows: normalizing flows in PyTorch

Torchflows is a library for generative modeling and density estimation using normalizing flows.
It implements many normalizing flow architectures and their building blocks for:

* easy use of normalizing flows as trainable distributions;
* easy implementation of new normalizing flows.

Example use:

```python
import torch
from torchflows.flows import Flow
from torchflows.architectures import RealNVP

torch.manual_seed(0)

n_data = 1000
n_dim = 3

x = torch.randn(n_data, n_dim)  # Generate some training data
bijection = RealNVP(n_dim)  # Create the bijection
flow = Flow(bijection)  # Create the normalizing flow

flow.fit(x)  # Fit the normalizing flow to training data
log_prob = flow.log_prob(x)  # Compute the log probability of training data
x_new = flow.sample(50)  # Sample 50 new data points

print(log_prob.shape)  # (100,)
print(x_new.shape)  # (50, 3)
```

Check examples and documentation, including the list of supported architectures [here](https://torchflows.readthedocs.io/en/latest/).
We also provide examples [here](examples/).

## Installing

We support Python versions 3.7 and upwards.

Install Torchflows via pip:

```
pip install torchflows
```

Install Torchflows directly from Github:

```
pip install git+https://github.com/davidnabergoj/torchflows.git
```

Setup for development:

```
git clone https://github.com/davidnabergoj/torchflows.git
cd torchflows
pip install -r requirements.txt
```


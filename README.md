# Torchflows: normalizing flows in PyTorch

Torchflows is a library for generative modeling and density estimation using normalizing flows.
It implements many normalizing flow architectures and their building blocks for:

* Easy use of normalizing flows as trainable distributions.
* Easy implementation of new normalizing flows.

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

Check out [examples](examples/) and the [documentation](https://torchflows.readthedocs.io/en/latest/), including the list of [supported architectures](https://torchflows.readthedocs.io/en/latest/).

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

## Citation

If you use this code in your work, we kindly ask that you cite the accompanying paper:
> [Nabergoj and Štrumbelj: Empirical evaluation of normalizing flows in Markov Chain Monte Carlo, 2024. arxiv:2412.17136.](https://arxiv.org/abs/2412.17136)

BibTex entry:
```
@misc{nabergoj_nf_mcmc_evaluation_2024,
    author = {Nabergoj, David and \v{S}trumbelj, Erik},
    title = {Empirical evaluation of normalizing flows in {Markov} {Chain} {Monte} {Carlo}},
    publisher = {arXiv},
    month = dec,
    year = {2024},
    note = {arxiv:2412.17136}
}
```

## Contributions

We warmly welcome all contributions and comments. 
Please do not hesitate to submit [issues](https://github.com/davidnabergoj/torchflows/issues) and [pull requests](https://github.com/davidnabergoj/torchflows/pulls).

Some options to start contributing include:
* Adding references to the documentation page for [architecture presets](https://torchflows.readthedocs.io/en/latest/architectures/index.html).
* Implementing new normalizing flow architectures (see the [developer guide](https://torchflows.readthedocs.io/en/latest/developer_reference/index.html)).
* Adding more [automated tests](./test) for numerical stability and optimization.
* Adding docstrings to undocumented classes.
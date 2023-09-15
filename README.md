# Normalizing flows in PyTorch

This package implements normalizing flows and their building blocks.

Example use:

```python
import torch
from normalizing_flows import RealNVP, Flow

torch.manual_seed(0)

n_data = 100
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

## Brief background

A normalizing flow (NF) is a flexible distribution, defined as a bijective transformation of a simple statistical
distribution.
The simple distribution is typically a standard Gaussian.
The transformation is typically an invertible neural network that can make the NF arbitrarily complex.
Training a NF using a dataset means optimizing the parameters of transformation to make the dataset likely under the NF.
We can use a NF to compute the probability of a data point or to independently sample data from the process that
generated our dataset.

A NF $q(x)$ with the bijection $f(z) = x$ and base distribution $p(z)$ is defined as:
$$\log q(x) = \log p(f^{-1}(x)) + \log\left|\det J_{f^{-1}}(x)\right|$$

## Implemented architectures

We implement the following NF transformations:

| Bijection              |   Inverse   |    Log determinant     | Inverse implemented |
|------------------------|:-----------:|:----------------------:|:-------------------:|
| NICE                   |    Exact    |         Exact          |         Yes         |
| Real NVP               |    Exact    |         Exact          |         Yes         |
| MAF                    |    Exact    |         Exact          |         Yes         |
| IAF                    |    Exact    |         Exact          |         Yes         |
| Rational quadratic NSF |    Exact    |         Exact          |         Yes         |
| Linear rational NSF    |    Exact    |         Exact          |         Yes         |
| UMNN flows             |             |                        |                     |
| Planar                 | Approximate |         Exact          |         No          |
| Radial                 | Approximate |         Exact          |         No          |
| Sylvester              | Approximate |         Exact          |         No          |
| Invertible ResNet      | Approximate |  Biased approximation  |         Yes         |
| ResFlow                | Approximate | Unbiased approximation |         Yes         |
| Proximal ResFlow       |             |                        |                     |
| Quasi AR flow          |             |                        |                     |
| FFJORD                 |             |      Approximate       |                     |
| RNode                  |             |      Approximate       |                     |
| DDNF                   |             |      Approximate       |                     |
| OT flow                |             |         Exact          |                     |

Architectures without an implemented inverse support either sampling or density estimation, but not both at once.
Such architectures are unsuitable for downstream tasks which require both functionalities.

We also implement simple bijections that can be used in the same manner:

* Permutation
* Elementwise translation (shift vector)
* Elementwise scaling (diagonal matrix)
* Rotation (orthogonal matrix)
* Triangular matrix
* Dense matrix (using the QR or LU decomposition)

All of these have exact inverses and log determinants.
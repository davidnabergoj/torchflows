# Normalizing flows in PyTorch

This package implements normalizing flows and their building blocks.
The package is meant for researchers, enabling:

* easy use of normalizing flows as generative models or density estimators in various applications;
* systematic comparisons of normalizing flows or their building blocks;
* simple implementation of new normalizing flows which belong to either the autoregressive, residual, or continuous
  families;

Example use:

```python
import torch
from normalizing_flows import RealNVP, Flow

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

We provide more examples [here](examples/README.md).

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

| Bijection                                                           |   Inverse   |     Log determinant     | Inverse implemented |
|---------------------------------------------------------------------|:-----------:|:-----------------------:|:-------------------:|
| [NICE](http://arxiv.org/abs/1410.8516)                              |    Exact    |          Exact          |         Yes         |
| [Real NVP](http://arxiv.org/abs/1605.08803)                         |    Exact    |          Exact          |         Yes         |
| [MAF](http://arxiv.org/abs/1705.07057)                              |    Exact    |          Exact          |         Yes         |
| [IAF](http://arxiv.org/abs/1606.04934)                              |    Exact    |          Exact          |         Yes         |
| [Rational quadratic NSF](http://arxiv.org/abs/1906.04032)           |    Exact    |          Exact          |         Yes         |
| [Linear rational NSF](http://arxiv.org/abs/2001.05168)              |    Exact    |          Exact          |         Yes         |
| [NAF](http://arxiv.org/abs/1804.00779)                              |             |                         |                     |
| [Block NAF](http://arxiv.org/abs/1904.04676)                        |             |                         |                     |
| [UMNN](http://arxiv.org/abs/1908.05164)                             | Approximate |          Exact          |         No          |
| [Planar](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423) | Approximate |          Exact          |         No          |
| [Radial](https://proceedings.mlr.press/v37/rezende15.html)          | Approximate |          Exact          |         No          |
| [Sylvester](http://arxiv.org/abs/1803.05649)                        | Approximate |          Exact          |         No          |
| [Invertible ResNet](http://arxiv.org/abs/1811.00995)                | Approximate |  Biased approximation   |         Yes         |
| [ResFlow](http://arxiv.org/abs/1906.02735)                          | Approximate | Unbiased approximation  |         Yes         |
| [Proximal ResFlow](http://arxiv.org/abs/2211.17158)                 | Approximate | Exact (if single layer) |         Yes         |
| [FFJORD](http://arxiv.org/abs/1810.01367)                           | Approximate |       Approximate       |         Yes         |
| [RNODE](http://arxiv.org/abs/2002.02798)                            | Approximate |       Approximate       |         Yes         |
| [DDNF](http://arxiv.org/abs/1810.03256)                             | Approximate |       Approximate       |         Yes         |
| [OT flow](http://arxiv.org/abs/2006.00104)                          | Approximate |          Exact          |         Yes         |

Note: inverse approximations can be made arbitrarily accurate with stricter convergence conditions.
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
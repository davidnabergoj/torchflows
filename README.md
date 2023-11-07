# Normalizing flows in PyTorch

This package implements normalizing flows and their building blocks.
It allows:

* easy use of normalizing flows as trainable distributions;
* easy implementation of new normalizing flows.

Example use:

```python
import torch
from normalizing_flows import Flow
from normalizing_flows.architectures import RealNVP


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

We provide more examples [here](examples/).

## Installing

Install the package:

```
pip install git+https://github.com/davidnabergoj/normalizing-flows.git
```

Setup for development:

```
git clone https://github.com/davidnabergoj/normalizing-flows.git
cd normalizing-flows
pip install -r requirements.txt
```

We support Python versions 3.7 and upwards.

## Brief background

A normalizing flow (NF) is a flexible trainable distribution.
It is defined as a bijective transformation of a simple distribution, such as a standard Gaussian.
The bijection is typically an invertible neural network.
Training a NF using a dataset means optimizing the bijection's parameters to make the dataset likely under the NF.
We can use a NF to compute the probability of a data point or to independently sample data from the process that
generated our dataset.

The density of a NF $q(x)$ with the bijection $f(z) = x$ and base distribution $p(z)$ is defined as:
$$\log q(x) = \log p(f^{-1}(x)) + \log\left|\det J_{f^{-1}}(x)\right|.$$
Sampling from a NF means sampling from the simple distribution and transforming the sample using the bijection.

## Supported architectures

We list supported NF architectures below.
We classify architectures as either autoregressive, residual, or continuous; as defined
by [Papamakarios et al. (2021)](https://arxiv.org/abs/1912.02762).
Exact architectures do not use numerical approximations to generate data or compute the log density.

| Architecture                                                           	 | Bijection type           	 | Exact 	 | Two-way |
|--------------------------------------------------------------------------|:--------------------------:|:-------:|:-------:|
| [NICE](http://arxiv.org/abs/1410.8516)                              	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [Real NVP](http://arxiv.org/abs/1605.08803)                         	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [MAF](http://arxiv.org/abs/1705.07057)                              	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [IAF](http://arxiv.org/abs/1606.04934)                              	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [Rational quadratic NSF](http://arxiv.org/abs/1906.04032)           	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [Linear rational NSF](http://arxiv.org/abs/2001.05168)              	    |      Autoregressive 	      | ✔     	 |    ✔    |
| [NAF](http://arxiv.org/abs/1804.00779)                              	    |      Autoregressive 	      | ✗     	 |    ✔    |
| [UMNN](http://arxiv.org/abs/1908.05164)                             	    |      Autoregressive 	      | ✗     	 |    ✔    |
| [Planar](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423) 	    |      Residual       	      | ✗     	 |    ✗    |
| [Radial](https://proceedings.mlr.press/v37/rezende15.html)          	    |      Residual       	      | ✗     	 |    ✗    |
| [Sylvester](http://arxiv.org/abs/1803.05649)                        	    |      Residual       	      | ✗     	 |    ✗    |
| [Invertible ResNet](http://arxiv.org/abs/1811.00995)                	    |      Residual       	      | ✗     	 |   ✔*    |
| [ResFlow](http://arxiv.org/abs/1906.02735)                          	    |      Residual       	      | ✗     	 |   ✔*    |
| [Proximal ResFlow](http://arxiv.org/abs/2211.17158)                 	    |      Residual       	      | ✗     	 |   ✔*    |
| [FFJORD](http://arxiv.org/abs/1810.01367)                           	    |      Continuous     	      | ✗     	 |   ✔*    |
| [RNODE](http://arxiv.org/abs/2002.02798)                            	    |      Continuous     	      | ✗     	 |   ✔*    |
| [DDNF](http://arxiv.org/abs/1810.03256)                             	    |      Continuous     	      | ✗     	 |   ✔*    |
| [OT flow](http://arxiv.org/abs/2006.00104)                          	    |      Continuous     	      | ✗     	 |    ✔    |

Two-way architectures support both sampling and density estimation.
Two-way architectures marked with an asterisk (*) support both, but use a numerical approximation to sample or estimate
density.
One-way architectures support either sampling or density estimation, but not both at once.

We also support simple bijections (all exact and two-way):

* Permutation
* Elementwise translation (shift vector)
* Elementwise scaling (diagonal matrix)
* Rotation (orthogonal matrix)
* Triangular matrix
* Dense matrix (using the QR or LU decomposition)

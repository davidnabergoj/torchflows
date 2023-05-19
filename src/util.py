import torch


def base_softplus(x: torch.Tensor):
    # Numerically stable implementation of softplus(x) = log(1 + exp(x))
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


# def softplus(x: torch.Tensor, limit=50):
#     # Even more stable implementation of softplus. We use the fact that log(1+exp(x)) \approx log(exp(x)) = x for big x.
#     y = x
#     y[x < limit] = base_softplus(x[x < limit])
#     return y

def softplus(x: torch.Tensor):
    # Concise and apparently stable: https://stackoverflow.com/a/76007579
    return torch.logaddexp(torch.zeros_like(x), x)

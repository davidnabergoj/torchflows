# Test the speed of standard NF operations

import torch
import timeit
import matplotlib.pyplot as plt

from normalizing_flows import Flow
from normalizing_flows.architectures import (
    NICE,
    RealNVP,
    MAF,
    IAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF,
    InverseAutoregressiveRQNSF,
    CouplingLRS,
    MaskedAutoregressiveLRS,
    CouplingDSF,
    UMNNMAF,
    DeepDiffeomorphicBijection,
    RNODE,
    FFJORD,
    OTFlow,
    ResFlow,
    ProximalResFlow,
    InvertibleResNet
)


def avg_eval_time(flow: Flow, n_repeats: int = 30):
    total_time = timeit.timeit(lambda: flow.log_prob(x), number=n_repeats)
    return total_time / n_repeats


def avg_sampling_time(flow: Flow, batch_size: int = 100, n_repeats: int = 30):
    total_time = timeit.timeit(lambda: flow.sample(batch_size), number=n_repeats)
    return total_time / n_repeats


if __name__ == '__main__':
    torch.manual_seed(0)
    batch_shape = (100,)
    event_shape = (50,)
    x = torch.randn(*batch_shape, *event_shape)

    eval_times = {}
    sample_times = {}
    for bijection_class in [
        NICE,
        RealNVP,
        MAF,
        IAF,
        CouplingRQNSF,
        MaskedAutoregressiveRQNSF,
        InverseAutoregressiveRQNSF,
        CouplingLRS,
        MaskedAutoregressiveLRS,
        CouplingDSF,
        # UMNNMAF,  # Too slow
        DeepDiffeomorphicBijection,
        RNODE,
        FFJORD,
        OTFlow,
        ResFlow,
        ProximalResFlow,
        InvertibleResNet
    ]:
        f = Flow(bijection_class(event_shape))

        name = bijection_class.__name__
        e_avg = avg_eval_time(f)
        s_avg = avg_sampling_time(f)

        print(f'{name:<30}\t| e: {e_avg:.4f}\t| s: {s_avg:.4f}')
        eval_times[name] = e_avg
        sample_times[name] = s_avg

    plt.figure()
    plt.bar(list(eval_times.keys()), list(eval_times.values()))
    plt.ylabel("log_prob time [s]")
    plt.xlabel("Bijection")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(list(sample_times.keys()), list(sample_times.values()))
    plt.ylabel("Sampling time [s]")
    plt.xlabel("Bijection")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

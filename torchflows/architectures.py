from torchflows.bijections.finite.autoregressive.architectures import (
    NICE,
    RealNVP,
    InverseRealNVP,
    MAF,
    IAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF,
    InverseAutoregressiveRQNSF,
    CouplingLRS,
    MaskedAutoregressiveLRS,
    InverseAutoregressiveLRS,
    CouplingDeepSF,
    MaskedAutoregressiveDeepSF,
    InverseAutoregressiveDeepSF,
    CouplingDenseSF,
    MaskedAutoregressiveDenseSF,
    InverseAutoregressiveDenseSF,
    CouplingDeepDenseSF,
    MaskedAutoregressiveDeepDenseSF,
    InverseAutoregressiveDeepDenseSF,
    UMNNMAF
)

from torchflows.bijections.continuous.ddnf import DeepDiffeomorphicBijection
from torchflows.bijections.continuous.rnode import RNODE
from torchflows.bijections.continuous.ffjord import FFJORD
from torchflows.bijections.continuous.otflow import OTFlow

from torchflows.bijections.finite.residual.architectures import (
    ResFlow,
    ProximalResFlow,
    InvertibleResNet,
    PlanarFlow,
    RadialFlow,
    SylvesterFlow
)

from torchflows.bijections.finite.multiscale.architectures import (
    MultiscaleRealNVP,
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    # MultiscaleDeepSigmoid,  # TODO stabler initialization
    # MultiscaleDenseSigmoid,  # TODO stabler initialization
    # MultiscaleDeepDenseSigmoid  # TODO stabler initialization
)

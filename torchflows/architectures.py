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

from torchflows.bijections.continuous.ddnf import DeepDiffeomorphicBijection, ConvolutionalDeepDiffeomorphicBijection
from torchflows.bijections.continuous.rnode import RNODE, ConvolutionalRNODE
from torchflows.bijections.continuous.ffjord import FFJORD, ConvolutionalFFJORD
from torchflows.bijections.continuous.otflow import OTFlow

from torchflows.bijections.finite.residual.architectures import (
    ResFlow,
    ProximalResFlow,
    InvertibleResNet,
    PlanarFlow,
    RadialFlow,
    SylvesterFlow,
    ConvolutionalInvertibleResNet,
    ConvolutionalResFlow
)

from torchflows.bijections.finite.multiscale.architectures import (
    MultiscaleRealNVP,
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    MultiscaleDeepSigmoid,
    MultiscaleDenseSigmoid,
    MultiscaleDeepDenseSigmoid,
    AffineGlow,
    RQSGlow,
    LRSGlow,
    ShiftGlow
)

from normalizing_flows.bijections.finite.residual.base import ResidualComposition
from normalizing_flows.bijections.finite.residual.iterative import InvertibleResNetBlock, ResFlowBlock
from normalizing_flows.bijections.finite.residual.proximal import ProximalResFlowBlock


class InvertibleResNet(ResidualComposition):
    def __init__(self, event_shape, context_shape, n_layers: int = 64, **kwargs):
        blocks = [
            InvertibleResNetBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)


class ResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape, n_layers: int = 64, **kwargs):
        blocks = [
            ResFlowBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)


class ProximalResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 64, **kwargs):
        blocks = [
            ProximalResFlowBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)

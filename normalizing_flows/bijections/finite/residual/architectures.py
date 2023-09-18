from normalizing_flows.bijections.finite.residual.base import ResidualComposition
from normalizing_flows.bijections.finite.residual.iterative import InvertibleResNetBlock, ResFlowBlock
from normalizing_flows.bijections.finite.residual.proximal import ProximalResFlowBlock


class InvertibleResNet(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        block = InvertibleResNetBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
        blocks = [block for _ in range(n_layers)]  # The same block
        super().__init__(blocks)


class ResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        block = ResFlowBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
        blocks = [block for _ in range(n_layers)]  # The same block
        super().__init__(blocks)


class ProximalResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        block = ProximalResFlowBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
        blocks = [block for _ in range(n_layers)]  # The same block
        super().__init__(blocks)

from normalizing_flows.bijections.finite.multiscale.base import MultiscaleBijection
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.affine import Scale, Affine, Shift


class MultiscaleAffineCoupling(MultiscaleBijection):
    def __init__(self, input_event_shape, **kwargs):
        super().__init__(input_event_shape, transformer_class=Affine, **kwargs)

from normalizing_flows.bijections.finite.autoregressive.layers import ElementwiseAffine
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.affine import Affine
from normalizing_flows.bijections import BijectiveComposition
from normalizing_flows.bijections.finite.multiscale.base import MultiscaleBijection


def make_image_layers(event_shape,
                      transformer_class,
                      n_layers: int = 2):
    """
    Returns a list of bijections for transformations of images with multiple channels.
    """
    if len(event_shape) != 3:
        raise ValueError("Multichannel image transformation are only possible for inputs with three axes.")

    assert n_layers >= 1

    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers - 1):
        bijections.append(
            MultiscaleBijection(
                input_event_shape=bijections[-1].transformed_shape,
                transformer_class=transformer_class
            )
        )
    bijections.append(
        MultiscaleBijection(
            input_event_shape=bijections[-1].transformed_shape,
            transformer_class=transformer_class,
            n_checkerboard_layers=4,
            squeeze_layer=False,
            n_channel_wise_layers=0
        )
    )
    bijections.append(ElementwiseAffine(event_shape=bijections[-1].transformed_shape))
    return bijections


class MultiscaleRealNVP(BijectiveComposition):
    def __init__(self,
                 event_shape,
                 n_layers: int = 3,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_image_layers(event_shape, Affine, n_layers)
        super().__init__(event_shape, bijections, **kwargs)
        self.transformed_shape = bijections[-1].transformed_shape

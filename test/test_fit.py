import pytest
import torch
from torchflows import Flow
from test.constants import __test_constants
from torchflows.bijections.finite.autoregressive.architectures import (
    NICE,
    RealNVP,
    MAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF
)
from torchflows.bijections.finite.autoregressive.layers import (
    ElementwiseScale,
    ElementwiseAffine,
    ElementwiseShift,
    ElementwiseRQSpline
)
from torchflows.bijections.finite.matrix import (
    LowerTriangularInvertibleMatrix,
    LUMatrix,
    QRMatrix
)
from torchflows.bijections.continuous.rnode import RNODE
from torchflows.bijections.finite.residual.architectures import ResFlow


@pytest.mark.local_only
def test_diagonal_gaussian_elementwise_affine():
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = ElementwiseAffine(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=100)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


@pytest.mark.local_only
def test_diagonal_gaussian_elementwise_scale():
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = ElementwiseScale(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=250, lr=0.1)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)

    print(x_std)

    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


@pytest.mark.local_only
@pytest.mark.parametrize('bijection_class',
                         [
                             LowerTriangularInvertibleMatrix,
                             LUMatrix,
                             QRMatrix,
                             MaskedAutoregressiveRQNSF,
                             # ElementwiseRQSpline,
                             ElementwiseAffine,
                             RealNVP,
                             MAF,
                             CouplingRQNSF,
                             ResFlow,
                             # RNODE  # takes a very long time, but goes towards correct std
                         ])
@pytest.mark.parametrize('target',
                         [
                             'diagonal',
                             'standard'
                         ])
def test_gaussian(bijection_class, target):
    if target == 'diagonal':
        sigma = torch.tensor([[0.1, 1.0, 10.0]])
    else:
        sigma = torch.tensor([[1.0, 1.0, 1.0]])

    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    x = torch.randn(size=(n_data, n_dim)) * sigma

    n_epochs = 150
    rtol = 0.1

    # Create flow
    if bijection_class in [LowerTriangularInvertibleMatrix]:
        bijection = bijection_class(event_shape=(n_dim,))
        n_epochs = 100
    elif bijection_class in [ElementwiseRQSpline]:
        bijection = bijection_class(
            event_shape=(n_dim,), n_bins=3, boundary=250.0)
        n_epochs = 1000
        rtol = 0.25
    elif bijection_class in [RNODE]:
        bijection = bijection_class(
            event_shape=(n_dim,),
            nn_kwargs={
                'hidden_size': 10 * n_dim,
                'n_hidden_layers': 1
            }
        )
        n_epochs = 5000
        rtol = 0.25
    else:
        bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)

    # Fit flow
    flow.fit(x, n_epochs=n_epochs)

    # Estimate std
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < rtol, f"Estimated std: {x_std}"


@pytest.mark.parametrize("n_train", [1, 10, 2200])
@pytest.mark.parametrize("event_shape", __test_constants["event_shape"])
def test_fit_basic(n_train, event_shape):
    torch.manual_seed(0)
    x_train = torch.randn(size=(n_train, *event_shape))
    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, n_epochs=2)


@pytest.mark.parametrize("n_train", [1, 10, 4000])
@pytest.mark.parametrize("n_val", [1, 10, 2400])
def test_fit_with_validation_data(n_train, n_val):
    torch.manual_seed(0)

    event_shape = (2, 3)

    x_train = torch.randn(size=(n_train, *event_shape))
    x_val = torch.randn(size=(n_val, *event_shape))

    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, n_epochs=2, x_val=x_val)


@pytest.mark.parametrize("n_train", [1, 10, 2200])
@pytest.mark.parametrize("event_shape", __test_constants["event_shape"])
@pytest.mark.parametrize("context_shape", __test_constants["context_shape"])
def test_fit_with_training_context(n_train, event_shape, context_shape):
    torch.manual_seed(0)
    x_train = torch.randn(size=(n_train, *event_shape))
    c_train = torch.randn(size=(n_train, *context_shape)) if context_shape is not None else None
    flow = Flow(RealNVP(event_shape, context_shape=context_shape))
    flow.fit(x_train, n_epochs=2, context_train=c_train)


@pytest.mark.parametrize("n_train", [1, 10, 2200])
@pytest.mark.parametrize("n_val", [1, 10, 2200])
@pytest.mark.parametrize("event_shape", __test_constants["event_shape"])
@pytest.mark.parametrize("context_shape", __test_constants["context_shape"])
def test_fit_with_context_and_validation_data(n_train, n_val, event_shape, context_shape):
    torch.manual_seed(0)

    # Setup training data
    x_train = torch.randn(size=(n_train, *event_shape))
    c_train = torch.randn(size=(n_train, *context_shape)) if context_shape is not None else None

    # Setup validation data
    x_val = torch.randn(size=(n_val, *event_shape))
    c_val = torch.randn(size=(n_val, *context_shape)) if context_shape is not None else None

    flow = Flow(RealNVP(event_shape, context_shape=context_shape))
    flow.fit(x_train, n_epochs=2, context_train=c_train, x_val=x_val, context_val=c_val)

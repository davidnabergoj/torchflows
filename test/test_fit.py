import pytest
import torch
from normalizing_flows import Flow
from normalizing_flows.bijections import NICE, RealNVP, MAF, ElementwiseAffine, ElementwiseShift, ElementwiseRQSpline, \
    CouplingRQNSF, MaskedAutoregressiveRQNSF, LowerTriangular, ElementwiseScale, QR, LU
from test.constants import __test_constants


@pytest.mark.skip(reason='Takes too long, fit quality is architecture-dependent')
@pytest.mark.parametrize('bijection_class', [
    LowerTriangular,
    ElementwiseScale,
    LU,
    QR,
    ElementwiseAffine,
    ElementwiseShift,
    ElementwiseRQSpline,
    NICE,
    RealNVP,
    MAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF
])
def test_standard_gaussian(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    x = torch.randn(size=(n_data, n_dim))
    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=15)
    x_flow = flow.sample(100_000)
    x_mean = torch.mean(x_flow, dim=0)
    x_var = torch.var(x_flow, dim=0)

    assert torch.allclose(x_mean, torch.zeros(size=(n_dim,)), atol=0.1)
    assert torch.allclose(x_var, torch.ones(size=(n_dim,)), atol=0.1)


@pytest.mark.skip(reason='Takes too long, fit quality is architecture-dependent')
def test_diagonal_gaussian_elementwise_affine():
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = ElementwiseAffine(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=15)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


@pytest.mark.skip(reason='Takes too long, fit quality is architecture-dependent')
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


@pytest.mark.skip(reason='Takes too long, fit quality is architecture-dependent')
@pytest.mark.parametrize('bijection_class',
                         [
                             LowerTriangular,
                             LU,
                             QR,
                             MaskedAutoregressiveRQNSF,
                             ElementwiseRQSpline,
                             ElementwiseAffine,
                             RealNVP,
                             MAF,
                             CouplingRQNSF
                         ])
def test_diagonal_gaussian_1(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)
    if isinstance(bijection, LowerTriangular):
        flow.fit(x, n_epochs=100)
    else:
        flow.fit(x, n_epochs=25)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


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
    if context_shape is None:
        c_train = None
    else:
        c_train = torch.randn(size=(n_train, *context_shape))
    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, n_epochs=2, context_train=c_train)


@pytest.mark.parametrize("n_train", [1, 10, 2200])
@pytest.mark.parametrize("n_val", [1, 10, 2200])
@pytest.mark.parametrize("event_shape", __test_constants["event_shape"])
@pytest.mark.parametrize("context_shape", __test_constants["context_shape"])
def test_fit_with_context_and_validation_data(n_train, n_val, event_shape, context_shape):
    torch.manual_seed(0)

    # Setup training data
    x_train = torch.randn(size=(n_train, *event_shape))
    if context_shape is None:
        c_train = None
    else:
        c_train = torch.randn(size=(n_train, *context_shape))

    # Setup validation data
    x_val = torch.randn(size=(n_val, *event_shape))
    if context_shape is None:
        c_val = None
    else:
        c_val = torch.randn(size=(n_val, *context_shape))

    flow = Flow(RealNVP(event_shape))
    flow.fit(x_train, n_epochs=2, context_train=c_train, x_val=x_val, context_val=c_val)

import pytest
import torch
from torchflows.bijections import RealNVP
from torchflows import Flow


@pytest.mark.skip(reason="Too slow on CI/CD")
def test_real_nvp_log_prob_data_on_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.log_prob(x_train)


@pytest.mark.skip(reason="Too slow on CI/CD")
def test_real_nvp_log_prob_data_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.log_prob(x_train.cuda())


@pytest.mark.skip(reason="Too slow on CI/CD")
def test_real_nvp_fit_data_on_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train)


@pytest.mark.skip(reason="Too slow on CI/CD")
def test_real_nvp_fit_data_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train.cuda())

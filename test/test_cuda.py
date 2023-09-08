import pytest
import torch
from normalizing_flows.bijections import RealNVP
from normalizing_flows import Flow


def test_real_nvp_log_prob_data_on_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.log_prob(x_train)


def test_real_nvp_log_prob_data_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.log_prob(x_train.cuda())


def test_real_nvp_fit_data_on_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train)


def test_real_nvp_fit_data_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train.cuda())

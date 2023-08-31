import torch
from normalizing_flows.bijections import RealNVP
from normalizing_flows import Flow


def test_real_nvp_fit_data_on_cpu():
    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train)


def test_real_nvp_fit_data_on_gpu():
    torch.manual_seed(0)

    batch_shape = (3, 5)
    event_shape = (7, 11)

    x_train = torch.randn(*batch_shape, *event_shape)

    flow = Flow(RealNVP(event_shape)).cuda()
    flow.fit(x_train.cuda())

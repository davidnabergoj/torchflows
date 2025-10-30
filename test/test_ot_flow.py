import torch
import pytest

from torchflows.bijections.continuous.otflow import OTFlowBijection, OTFlowTimeDerivative, OTPotential, OTResNet, concatenate_x_t

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (5,)])
def test_concatenate_x_t_valid_shape(batch_size, event_shape):
    torch.manual_seed(0)
    x = torch.zeros(size=(batch_size, *event_shape))
    t = torch.ones(size=())
    c = concatenate_x_t(x, t)
    assert c.shape == (batch_size, event_shape[0] + 1)
    assert torch.all(c[..., :-1] == x)
    assert torch.all(c[..., -1] == t)

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_shape', [(1, 2), (2, 4)])
def test_concatenate_x_t_invalid_x_shape(batch_size, event_shape):
    torch.manual_seed(0)
    x = torch.zeros(size=(batch_size, *event_shape))
    t = torch.ones(size=())
    with pytest.raises(ValueError):
        concatenate_x_t(x, t)

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('t_shape', [(1,), (2,), (5,)])
def test_concatenate_x_t_invalid_t_shape(batch_size, t_shape):
    torch.manual_seed(0)
    x = torch.zeros(size=(batch_size, 10))
    t = torch.ones(size=t_shape)
    with pytest.raises(ValueError):
        concatenate_x_t(x, t)

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_resnet_forward(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    rn = OTResNet(
        c_event_size=event_size + 1, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    c = concatenate_x_t(x, t)
    rn_out = rn.forward(s=c)

    assert rn_out.shape == (batch_size, hidden_size)
    assert rn_out.isfinite().all()

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_resnet_compute_u0(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    rn = OTResNet(
        c_event_size=event_size + 1, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    c = concatenate_x_t(x, t)
    u0 = rn.compute_u0(s=c)

    assert u0.shape == (batch_size, hidden_size)
    assert u0.isfinite().all()

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_resnet_jvp(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    rn = OTResNet(
        c_event_size=event_size + 1, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    c = concatenate_x_t(x, t)
    w = torch.randn(size=(batch_size, hidden_size))
    jvp = rn.jvp(s=c, w=w)

    assert jvp.shape == c.shape
    assert jvp.isfinite().all()

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_resnet_hessian_trace(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    rn = OTResNet(
        c_event_size=event_size + 1, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    c = concatenate_x_t(x, t)
    w = torch.randn(size=(batch_size, hidden_size))

    u0 = rn.compute_u0(s=c)
    z1 = rn.compute_z1(w=w, u0=u0)

    tr = rn.hessian_trace(s=c, w=w, u0=u0, z1=z1)

    assert tr.shape == (batch_size,)
    assert tr.isfinite().all()


@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_potential_forward(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    potential = OTPotential(
        event_size=event_size, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    grad_u_x, grad_u_t = potential.forward(t, x)

    assert grad_u_x.shape == (batch_size, event_size)
    assert grad_u_x.isfinite().all()

    assert grad_u_t.shape == (batch_size, 1)
    assert grad_u_t.isfinite().all()


@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_size', [1, 2, 5])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_potential_divergence(batch_size, event_size, hidden_size, step_size):
    torch.manual_seed(0)
    potential = OTPotential(
        event_size=event_size, 
        hidden_size=hidden_size, 
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, event_size))
    t = torch.rand(size=())
    s = concatenate_x_t(x, t)
    u0 = potential.resnet.compute_u0(s)
    z1 = potential.resnet.compute_z1(w=potential.w, u0=u0)
    div = potential.compute_divergence(s=s, u0=u0, z1=z1)

    assert div.shape == (batch_size,)
    assert div.isfinite().all()

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (5,), (2, 3, 5)])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
def test_td_forward(batch_size, event_shape, hidden_size, step_size):
    torch.manual_seed(0)
    td = OTFlowTimeDerivative(
        event_shape=event_shape,
        hidden_size=hidden_size,
        step_size=step_size
    )
    x = torch.randn(size=(batch_size, *event_shape))
    t = torch.rand(size=())

    initial_state = td.prepare_initial_state(z0=x)
    delta_state = td.forward(t, initial_state)

    for element in delta_state:
        assert element.shape[0] == batch_size
        assert element.isfinite().all()

@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (5,), (2, 3, 5)])
@pytest.mark.parametrize('hidden_size', [1, 2, 5])
@pytest.mark.parametrize('step_size', [0.01, 0.1, 1.0])
@pytest.mark.parametrize('training', [False, True])
def test_odeint(batch_size, event_shape, hidden_size, step_size, training):
    torch.manual_seed(0)
    b = OTFlowBijection(
        event_shape=event_shape,
        time_derivative_kwargs=dict(
            hidden_size=hidden_size,
            step_size=step_size
        ),
        solver='dopri5'
    )
    if training:
        b.train()
    else:
        b.eval()

    x = torch.randn(size=(batch_size, *event_shape))

    z, log_det_forward = b.forward(x)
    xr, log_det_inverse = b.inverse(z)

    assert z.shape == (batch_size, *event_shape)
    assert z.isfinite().all()

    assert xr.shape == (batch_size, *event_shape)
    assert xr.isfinite().all()

    assert log_det_forward.shape == (batch_size,)
    assert log_det_forward.isfinite().all()

    assert log_det_inverse.shape == (batch_size,)
    assert log_det_inverse.isfinite().all()

    assert torch.allclose(x, xr, atol=1e-2)
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=2e-2)
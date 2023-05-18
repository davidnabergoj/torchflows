import pytest
import torch

from src.bijections.finite.autoregressive.architectures import RealNVP, MAF


@pytest.mark.parametrize('n_dim', [2, 10, 100])
def test_maf(n_dim):
    # MAF reconstruction errors are larger with fewer input dimensions
    torch.manual_seed(0)
    bijection = MAF(n_dim)

    x = torch.randn(size=(125, n_dim)) * 5
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    if not torch.isclose(
            reconstruction_error := (x - x_reconstructed).abs().max(),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(reconstruction_error) = }')

    if not torch.isclose(
            reconstruction_error := torch.linalg.norm(x - x_reconstructed),
            torch.zeros(1),
            atol=1e-3):
        print(f'{reconstruction_error = }')
        raise ValueError(f'{float(reconstruction_error) = }')

    if not torch.isclose(
            log_det_error := (log_det_forward + log_det_inverse).abs().max(),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(log_det_error) = }')

    if not torch.isclose(
            log_det_error := torch.linalg.norm(log_det_forward + log_det_inverse),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(log_det_error) = }')


@pytest.mark.parametrize('n_dim', [2, 10, 100])
def test_real_nvp(n_dim):
    torch.manual_seed(0)
    bijection = RealNVP(n_dim)

    x = torch.randn(size=(125, n_dim)) * 5
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    if not torch.isclose(
            reconstruction_error := (x - x_reconstructed).abs().max(),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(reconstruction_error) = }')

    if not torch.isclose(
            reconstruction_error := torch.linalg.norm(x - x_reconstructed),
            torch.zeros(1),
            atol=1e-3):
        print(f'{reconstruction_error = }')
        raise ValueError(f'{float(reconstruction_error) = }')

    if not torch.isclose(
            log_det_error := (log_det_forward + log_det_inverse).abs().max(),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(log_det_error) = }')

    if not torch.isclose(
            log_det_error := torch.linalg.norm(log_det_forward + log_det_inverse),
            torch.zeros(1),
            atol=1e-3):
        raise ValueError(f'{float(log_det_error) = }')

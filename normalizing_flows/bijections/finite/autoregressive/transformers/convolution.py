from typing import Union, Tuple
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer


def construct_kernels_plu(
        lower_elements: torch.Tensor,
        upper_elements: torch.Tensor,
        log_abs_diag: torch.Tensor,
        sign_diag: torch.Tensor,
        permutation: torch.Tensor,
        k: int,
        inverse: bool = False
):
    """
    :param lower_elements: (b, (k ** 2 - k) // 2)
    :param upper_elements: (b, (k ** 2 - k) // 2)
    :param log_abs_diag: (b, k)
    :param sign_diag: (b, k)
    :param permutation: (k, k)
    :param k: kernel length
    :param inverse:
    :return: kernels with shape (b, k, k)
    """

    assert lower_elements.shape == upper_elements.shape
    assert log_abs_diag.shape == sign_diag.shape
    assert permutation.shape == (k, k)
    assert len(log_abs_diag.shape) == 2
    assert len(lower_elements.shape) == 2
    assert lower_elements.shape[1] == (k ** 2 - k) // 2
    assert log_abs_diag.shape[1] == k

    batch_size = len(lower_elements)

    lower = torch.eye(k)[None].repeat(batch_size)
    lower_row_idx, lower_col_idx = torch.tril_indices(k, k, offset=-1)
    lower[:, lower_row_idx, lower_col_idx] = lower_elements

    upper = torch.einsum("ij,bj->bij", torch.eye(k), log_abs_diag.exp() * sign_diag)
    upper_row_idx, upper_col_idx = torch.triu_indices(k, k, offset=1)
    upper[:, upper_row_idx, upper_col_idx] = upper_elements

    if inverse:
        if log_abs_diag.dtype == torch.float64:
            lower_inv = torch.inverse(lower)
            upper_inv = torch.inverse(upper)
        else:
            lower_inv = torch.inverse(lower.double()).type(log_abs_diag.dtype)
            upper_inv = torch.inverse(upper.double()).type(log_abs_diag.dtype)
        kernels = torch.einsum("bij,bjk,kl->bil", upper_inv, lower_inv, permutation.T)
    else:
        kernels = torch.einsum("ij,bjk,bkl->bil", permutation, lower, upper)
    return kernels


class Invertible1x1Convolution(Transformer):
    """
    Invertible 1x1 convolution.

    TODO permutation may be unnecessary, maybe remove.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], kernel_length: int = 3):
        if len(event_shape) != 3:
            raise ValueError(
                f"InvertibleConvolution transformer only supports events with shape (height, width, channels)."
            )
        self.c, self.h, self.w = event_shape
        if kernel_length <= 0:
            raise ValueError(f"Expected kernel length to be positive, but got {kernel_length}")
        self.k = kernel_length
        self.sign_diag = torch.sign(torch.randn(self.k))
        self.permutation = torch.eye(self.k)[torch.randperm(self.k)]
        self.const = 1000
        super().__init__(event_shape)

    @property
    def n_parameters(self) -> int:
        return self.k ** 2

    @property
    def default_parameters(self) -> torch.Tensor:
        # Kernel matrix is identity (p=0,u=0,log_diag=0).
        # Some diagonal elements are negated according to self.sign_diag.
        # The matrix is then permuted.
        return torch.zeros(self.n_parameters)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        We parametrize K = PLU. The parameters h contain elements of L and U.
        There are k ** 2 such elements.

        x.shape == (batch_size, c, h, w)
        h.shape == (batch_size, k * k)

        We expect each kernel to be invertible.
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected x to have shape (batch_size, channels, height, width), but got {x.shape}")
        if len(h.shape) != 2:
            raise ValueError(f"Expected h.shape to be of length 2, but got {h.shape} with length {len(h.shape)}")
        if h.shape[1] != self.k * self.k:
            raise ValueError(
                f"Expected h to have shape (batch_size, kernel_height * kernel_width) = (batch_size, {self.k * self.k}),"
                f" but got {h.shape}"
            )

        h = self.default_parameters + h / self.const

        n_p_elements = (self.k ** 2 - self.k) // 2
        p_elements = h[..., :n_p_elements]
        u_elements = h[..., n_p_elements:n_p_elements * 2]
        diag_elements = h[..., n_p_elements * 2:]

        kernels = construct_kernels_plu(
            p_elements,
            u_elements,
            diag_elements,
            self.sign_diag,
            self.permutation,
            self.k,
            inverse=False
        )
        log_det = self.h * self.w * torch.log(torch.abs(torch.linalg.det(kernels)))  # (*batch_shape)

        # Reshape images to (1, b, c, h, w), reshape kernels to (b, 1, k, k)
        # This lets us convolve each image with its own kernel
        z = torch.conv2d(x[None], kernels[:, None], groups=self.c)[0]

        return z, log_det

    def inverse(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) != 4:
            raise ValueError(f"Expected x to have shape (batch_size, height, width, channels), but got {x.shape}")
        if len(h.shape) != 2:
            raise ValueError(f"Expected h.shape to be of length 2, but got {h.shape} with length {len(h.shape)}")
        if h.shape[1] != self.k * self.k:
            raise ValueError(
                f"Expected h to have shape (batch_size, kernel_height * kernel_width) = (batch_size, {self.k * self.k}),"
                f" but got {h.shape}"
            )

        h = self.default_parameters + h / self.const

        n_p_elements = (self.k ** 2 - self.k) // 2
        p_elements = h[..., :n_p_elements]
        u_elements = h[..., n_p_elements:n_p_elements * 2]
        diag_elements = h[..., n_p_elements * 2:]

        kernels = construct_kernels_plu(
            p_elements,
            u_elements,
            diag_elements,
            self.sign_diag,
            self.permutation,
            self.k,
            inverse=True
        )
        log_det = -self.h * self.w * torch.log(torch.abs(torch.linalg.det(kernels)))  # (*batch_shape)

        # Reshape images to (1, b, c, h, w), reshape kernels to (b, 1, k, k)
        # This lets us convolve each image with its own kernel
        z = torch.conv2d(x[None], kernels[:, None], groups=self.c)[0]

        return z, log_det

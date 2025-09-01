"""PyTorch utilities for higher-order gradient support.

This module provides a thin wrapper around max-pooling that retains the
computation graph required for second-order gradient calculations. The
original TensorFlow version registered a custom gradient for
``MaxPoolGrad``; in PyTorch we can achieve the same effect by building a
``torch.autograd.Function`` that explicitly constructs the graph during
the backward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MaxPool2dWithGrad(torch.autograd.Function):
    """Max-pooling operation with double-backward support.

    The forward pass behaves like :func:`torch.nn.functional.max_pool2d`.
    During the backward pass we reconstruct the operation under a graph so
    that higher-order derivatives (e.g. gradients of gradients) can be
    taken. Only parameters relevant to the pooling operation are stored so
    that autograd can recompute the forward pass as needed.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
    ) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        return F.max_pool2d(input, kernel_size, stride, padding, dilation)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (inp,) = ctx.saved_tensors
        kwargs = ctx.params

        # Re-run the forward pass under a graph so autograd can compute
        # gradients with respect to the input while retaining the graph for
        # potential higher-order derivatives.
        with torch.enable_grad():
            reinput = inp.detach().requires_grad_(True)
            output = F.max_pool2d(reinput, **kwargs)
        grad_input = torch.autograd.grad(
            output, reinput, grad_output, retain_graph=True, create_graph=True
        )[0]

        return grad_input, None, None, None, None


def max_pool2d(
    input: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Apply a max-pooling layer with double-backward support.

    This convenience wrapper exposes an interface similar to
    :func:`torch.nn.functional.max_pool2d` but routes the computation
    through :class:`MaxPool2dWithGrad` to keep the graph needed for higher
    order differentiation.
    """

    return MaxPool2dWithGrad.apply(input, kernel_size, stride, padding, dilation)


__all__ = ["max_pool2d", "MaxPool2dWithGrad"]


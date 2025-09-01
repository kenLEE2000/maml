"""Utility helpers used across the PyTorch MAML codebase.

This module was originally written for a TensorFlow implementation and has
been rewritten to use PyTorch primitives.  Only a small subset of the original
helpers are retained as they are useful for building convolutional networks and
computing common losses.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def get_images(
    paths: Sequence[str],
    labels: Sequence[int],
    nb_samples: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[int, str]]:
    """Collect image file paths for the provided class folders.

    Args:
        paths: Iterable of directory paths, one for each class.
        labels: Iterable of integer labels corresponding to ``paths``.
        nb_samples: If given, randomly sample this many files from each class
            directory.  Otherwise include all files.
        shuffle: Whether to shuffle the resulting list of ``(label, path)``
            tuples.

    Returns:
        A list of ``(label, image_path)`` tuples.
    """

    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images = [
        (label, os.path.join(path, image))
        for label, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    if shuffle:
        random.shuffle(images)
    return images


# -----------------------------------------------------------------------------
# Network helpers
# -----------------------------------------------------------------------------
def conv_block(
    inp: Tensor,
    cweight: Tensor,
    bweight: Tensor,
    *,
    activation=F.relu,
    use_max_pool: bool = True,
    max_pool_pad: int = 0,
    norm: str = "batch_norm",
    residual: bool = False,
) -> Tensor:
    """Apply convolution, normalization, non-linearity and optional max-pool.

    The function mirrors the behavior of the original TensorFlow version but is
    implemented using :mod:`torch.nn.functional` operations.  ``cweight`` and
    ``bweight`` are expected to be learnable parameters shaped for use with
    :func:`torch.nn.functional.conv2d`.
    """

    stride = 1 if use_max_pool else 2
    out = F.conv2d(inp, cweight, bweight, stride=stride, padding=cweight.shape[-1] // 2)
    out = normalize(out, activation=activation, norm=norm)
    if use_max_pool:
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=max_pool_pad)
    if residual:
        out = out + inp
    return out


def normalize(inp: Tensor, *, activation=F.relu, norm: str = "batch_norm") -> Tensor:
    """Apply normalization followed by an optional activation function."""

    if norm == "batch_norm":
        out = F.batch_norm(inp, running_mean=None, running_var=None, training=True)
    elif norm == "layer_norm":
        out = F.layer_norm(inp, inp.shape[1:])
    elif norm == "None":
        out = inp
    else:
        raise ValueError(f"Unrecognized norm '{norm}'")

    if activation is not None:
        out = activation(out)
    return out


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------
def mse(pred: Tensor, label: Tensor) -> Tensor:
    """Mean squared error loss."""
    return torch.mean((pred.view(-1) - label.view(-1)) ** 2)


def xent(pred: Tensor, label: Tensor) -> Tensor:
    """Cross-entropy loss that accepts one-hot or index targets."""
    if pred.shape == label.shape:
        target = label.argmax(dim=1)
    else:
        target = label
    return F.cross_entropy(pred, target)

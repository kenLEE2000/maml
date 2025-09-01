"""PyTorch data generation utilities for MAML.

This module provides a minimal sinusoid task generator used by the
PyTorch-based ``main.py`` script.  It replaces the original TensorFlow
implementation and returns tensors ready for model consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class SineBatch:
    """Container for a batch of sinusoid regression tasks."""

    inputa: torch.Tensor
    labela: torch.Tensor
    inputb: torch.Tensor
    labelb: torch.Tensor


class DataGenerator:
    """Generate batches of sinusoid regression tasks.

    Parameters
    ----------
    update_batch_size : int
        Number of samples used for the inner adaptation step (K-shot).
    meta_batch_size : int
        Number of tasks per meta-update.
    config : dict, optional
        Optional overrides for amplitude, phase, and input ranges.
    """

    def __init__(
        self,
        update_batch_size: int,
        meta_batch_size: int,
        config: Dict | None = None,
    ) -> None:
        self.k = update_batch_size
        self.meta_batch_size = meta_batch_size
        self.config = config or {}

        self.amp_range = self.config.get("amp_range", [0.1, 5.0])
        self.phase_range = self.config.get("phase_range", [0, np.pi])
        self.input_range = self.config.get("input_range", [-5.0, 5.0])
        self.dim_input = 1
        self.dim_output = 1

    def sample_batch(self) -> SineBatch:
        """Sample a meta-batch of sinusoid tasks."""

        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], size=[self.meta_batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], size=[self.meta_batch_size])
        xs = np.random.uniform(
            self.input_range[0],
            self.input_range[1],
            size=[self.meta_batch_size, self.k * 2, 1],
        )
        ys = amp[:, None, None] * np.sin(xs - phase[:, None, None])
        xa = torch.tensor(xs[:, : self.k], dtype=torch.float32)
        ya = torch.tensor(ys[:, : self.k], dtype=torch.float32)
        xb = torch.tensor(xs[:, self.k :], dtype=torch.float32)
        yb = torch.tensor(ys[:, self.k :], dtype=torch.float32)
        return SineBatch(xa, ya, xb, yb)


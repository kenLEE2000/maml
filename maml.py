"""PyTorch implementation of the MAML model.

This module defines a small fully connected network along with an
`adapt` helper that performs a single inner-loop gradient update.  It is
used by ``main.py`` for sinusoid regression experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class MAML(nn.Module):
    """Simple 3-layer network for 1D regression tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1, 40)
        self.layer2 = nn.Linear(40, 40)
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Forward pass with optional ``params`` dict.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, 1)``.
        params: Dict[str, torch.Tensor], optional
            Parameter dictionary produced by :meth:`adapt`.  If ``None`` the
            module's current parameters are used.
        """
        if params is None:
            params = dict(self.named_parameters())
        x = nn.functional.linear(x, params["layer1.weight"], params["layer1.bias"])
        x = torch.relu(x)
        x = nn.functional.linear(x, params["layer2.weight"], params["layer2.bias"])
        x = torch.relu(x)
        x = nn.functional.linear(x, params["layer3.weight"], params["layer3.bias"])
        return x

    def adapt(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Dict[str, torch.Tensor] | None = None,
        lr: float = 0.01,
        create_graph: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return parameters updated with one gradient step.

        Parameters
        ----------
        x, y: torch.Tensor
            Support set inputs and targets.
        params: Dict[str, torch.Tensor], optional
            Parameters to update; defaults to the module's own weights.
        lr: float
            Inner-loop learning rate.
        create_graph: bool
            Whether to retain the computation graph for higher-order
            gradients.
        """
        if params is None:
            params = dict(self.named_parameters())
        preds = self.forward(x, params)
        loss = nn.functional.mse_loss(preds, y)
        grads = torch.autograd.grad(loss, params.values(), create_graph=create_graph)
        updated = {name: param - lr * grad for (name, param), grad in zip(params.items(), grads)}
        return updated, loss

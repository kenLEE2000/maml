"""PyTorch implementation of MAML for sinusoid regression.

This script replaces the original TensorFlow based entry point and provides a
minimal example of training Model-Agnostic Meta-Learning (MAML) using
PyTorch.  The implementation focuses on the sinusoid regression task described
in the original MAML paper and mirrors the command line interface of the old
`main.py` when possible.

Example usage:

```
# train for a few iterations
python main.py --train --iterations 500

# evaluate a trained model
python main.py --test
```
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn, optim
import random


@dataclass
class SineBatch:
    """Container for a batch of sinusoid tasks."""

    inputa: torch.Tensor
    labela: torch.Tensor
    inputb: torch.Tensor
    labelb: torch.Tensor


class SineGenerator:
    """Generate batches of sinusoid regression tasks.

    Each task is defined by an amplitude and phase sampled uniformly from
    predefined ranges.  For each task we generate K points for adaptation and
    K points for evaluation (where K is ``update_batch_size``).
    """

    def __init__(self, update_batch_size: int, meta_batch_size: int):
        self.k = update_batch_size
        self.meta_batch_size = meta_batch_size

    def sample_batch(self) -> SineBatch:
        amp = np.random.uniform(0.1, 5.0, size=[self.meta_batch_size])
        phase = np.random.uniform(0, np.pi, size=[self.meta_batch_size])
        xs = np.random.uniform(
            -5.0, 5.0, size=[self.meta_batch_size, self.k * 2, 1]
        )
        ys = amp[:, None, None] * np.sin(xs - phase[:, None, None])
        xa = torch.tensor(xs[:, : self.k], dtype=torch.float32)
        ya = torch.tensor(ys[:, : self.k], dtype=torch.float32)
        xb = torch.tensor(xs[:, self.k :], dtype=torch.float32)
        yb = torch.tensor(ys[:, self.k :], dtype=torch.float32)
        return SineBatch(xa, ya, xb, yb)


class MAML(nn.Module):
    """Simple fully-connected network used for sinusoid regression."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1, 40)
        self.layer2 = nn.Linear(40, 40)
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
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
        """Perform one inner-loop update and return adapted parameters."""

        if params is None:
            params = dict(self.named_parameters())
        preds = self.forward(x, params)
        loss = nn.functional.mse_loss(preds, y)
        grads = torch.autograd.grad(loss, params.values(), create_graph=create_graph)
        updated = {
            name: param - lr * grad for (name, param), grad in zip(params.items(), grads)
        }
        return updated, loss


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAML().to(device)
    opt = optim.Adam(model.parameters(), lr=args.meta_lr)
    generator = SineGenerator(args.update_batch_size, args.meta_batch_size)

    for itr in range(args.iterations):
        batch = generator.sample_batch()
        inputa, labela, inputb, labelb = [
            t.to(device) for t in (batch.inputa, batch.labela, batch.inputb, batch.labelb)
        ]
        meta_loss = 0.0
        for task in range(args.meta_batch_size):
            params = None
            for _ in range(args.num_updates):
                params, _ = model.adapt(
                    inputa[task], labela[task], params, lr=args.update_lr
                )
            preds = model.forward(inputb[task], params)
            loss = nn.functional.mse_loss(preds, labelb[task])
            meta_loss += loss
        meta_loss /= args.meta_batch_size
        opt.zero_grad()
        meta_loss.backward()
        opt.step()

        if (itr + 1) % args.print_interval == 0:
            print(f"Iteration {itr + 1}: meta loss {meta_loss.item():.4f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"Model saved to {args.checkpoint}")


def test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAML().to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint {args.checkpoint} not found. Train a model first."
        )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    generator = SineGenerator(args.update_batch_size, args.meta_batch_size)
    batch = generator.sample_batch()
    inputa, labela, inputb, labelb = [
        t.to(device) for t in (batch.inputa, batch.labela, batch.inputb, batch.labelb)
    ]

    losses = []
    for task in range(args.meta_batch_size):
        params = None
        # no need to build higher order derivatives when evaluating
        for _ in range(args.num_updates):
            params, _ = model.adapt(
                inputa[task], labela[task], params, lr=args.update_lr, create_graph=False
            )
        preds = model.forward(inputb[task], params)
        loss = nn.functional.mse_loss(preds, labelb[task])
        losses.append(loss.item())

    print(f"Mean loss after adaptation: {np.mean(losses):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of MAML on sinusoid regression"
    )
    parser.add_argument("--meta_batch_size", type=int, default=25, help="tasks per meta-update")
    parser.add_argument("--update_batch_size", type=int, default=10, help="K for K-shot learning")
    parser.add_argument("--num_updates", type=int, default=1, help="inner gradient steps")
    parser.add_argument("--meta_lr", type=float, default=1e-3, help="meta learning rate")
    parser.add_argument("--update_lr", type=float, default=0.01, help="inner learning rate")
    parser.add_argument("--iterations", type=int, default=10000, help="number of meta-training iterations")
    parser.add_argument("--checkpoint", type=str, default="maml.pth", help="path to save model")
    parser.add_argument("--print_interval", type=int, default=100, help="iterations between prints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="run training")
    group.add_argument("--test", action="store_true", help="evaluate a saved model")
    args = parser.parse_args()

    # Ensure reproducibility for demonstration purposes
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.train:
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()


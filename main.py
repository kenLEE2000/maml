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

import numpy as np
import torch
from torch import nn, optim
import random

from data_generator import DataGenerator
from maml import MAML


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAML().to(device)
    opt = optim.Adam(model.parameters(), lr=args.meta_lr)
    generator = DataGenerator(args.update_batch_size, args.meta_batch_size)

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

    generator = DataGenerator(args.update_batch_size, args.meta_batch_size)
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


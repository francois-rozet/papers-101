#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from torch import Tensor
from tqdm import tqdm
from typing import *


def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([
                nn.Linear(a, b),
                nn.ELU(),
                nn.LayerNorm(b, elementwise_affine=False),
            ])

        super().__init__(*layers[:-2])


class FFF(nn.Module):
    def __init__(self, features: int, **kwargs):
        super().__init__()

        self.f = MLP(features, features, **kwargs)  # encoder
        self.g = MLP(features, features, **kwargs)  # decoder

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)

    def sample(self, z: Tensor) -> Tensor:
        return self.g(z)

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        I = I.expand(*x.shape, -1).movedim(-1, 0)

        with torch.enable_grad():
            x = x.clone().requires_grad_()
            z = self.f(x)

        jacobian = torch.autograd.grad(z, x, I, is_grads_batched=True)[0].movedim(0, -1)
        ladj = torch.linalg.slogdet(jacobian).logabsdet

        return log_normal(z) + ladj


class FFFLoss(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()

        self.f = f
        self.g = g

    def forward(self, x: Tensor, beta: float = 10.0, hutchinson: int = 1) -> Tensor:
        x = x.clone().requires_grad_()
        z = self.f(x)
        y = self.g(z)

        l_re = (y - x).square().sum(dim=-1).mean()

        if hutchinson > 1:
            v0 = torch.randn_like(x.expand(hutchinson, *x.shape))
            v1 = torch.autograd.grad(y, z, v0, retain_graph=True, is_grads_batched=True)[0]
            v2 = torch.autograd.grad(z, x, v1, create_graph=True, is_grads_batched=True)[0]
        else:
            v0 = torch.randn_like(x)
            v1 = torch.autograd.grad(y, z, v0, retain_graph=True)[0]
            v2 = torch.autograd.grad(z, x, v1, create_graph=True)[0]

        l_ml = -(v0 * v2).sum(dim=-1).mean() - log_normal(z).mean()

        z_ = torch.randn_like(x)
        l_re_ = (z_ - self.f(self.g(z_))).square().sum(dim=-1).mean()

        return l_ml + beta * (l_re + l_re_)


if __name__ == '__main__':
    flow = FFF(2, hidden_features=[64] * 2)

    # Training
    loss = FFFLoss(flow.f, flow.g)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    data, _ = make_moons(16384, noise=0.05)
    data = torch.from_numpy(data).float()

    for epoch in tqdm(range(16384), ncols=88):
        subset = torch.randint(0, len(data), (256,))
        x = data[subset]

        loss(x).backward()

        optimizer.step()
        optimizer.zero_grad()

    # Sampling
    with torch.no_grad():
        z = torch.randn(16384, 2)
        x = flow.sample(z)

    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.hist2d(*x.T, bins=64)
    plt.savefig('moons.pdf')

    # Log-likelihood
    log_p = flow.log_prob(data[:4])

    print(log_p)

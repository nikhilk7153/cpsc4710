\
"""
density_ratio.py

A simple density-ratio discriminator used to approximate:
  log p_policy(z) / p_human(z)

If you train a binary classifier D(z) that distinguishes policy vs human latents with *balanced minibatches*,
then the logit approximates the log density ratio up to a constant.
Taking the mean of logits over policy samples gives a batchwise KL-style estimate:
  KL(p_policy || p_human) ≈ E_{z~p_policy}[log_ratio(z)]

This matches your proposal's idea of a discriminator-based KL penalty in SAE latent space.

Practical note:
SARM latents are 65,536-D. This class expects you to pass in a *subset projection* (e.g., k=128 dims)
to keep memory/compute manageable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DensityRatioStats:
    loss: float
    acc: float


class DensityRatioDiscriminator(nn.Module):
    """
    Minimal MLP discriminator for density-ratio estimation in a low-dimensional latent subspace.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # logits


class DensityRatioTrainer:
    """
    Owns a discriminator + optimizer and exposes:
      - train_step(z_human, z_policy)
      - log_ratio(z_policy)
    """

    def __init__(
        self,
        dim: int,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = DensityRatioDiscriminator(dim=dim).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def log_ratio(self, z_policy: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample logits, interpreted as log density ratio (up to a constant).
        """
        self.model.eval()
        z_policy = z_policy.to(self.device)
        return self.model(z_policy)

    def train_step(self, z_human: torch.Tensor, z_policy: torch.Tensor) -> DensityRatioStats:
        """
        One SGD step on a balanced batch.
        Labels: human=0, policy=1
        """
        self.model.train()

        z_human = z_human.to(self.device)
        z_policy = z_policy.to(self.device)

        x = torch.cat([z_human, z_policy], dim=0)
        y = torch.cat(
            [torch.zeros(z_human.shape[0], device=self.device), torch.ones(z_policy.shape[0], device=self.device)],
            dim=0,
        )

        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            preds = (logits > 0).float()
            acc = (preds == y).float().mean().item()

        return DensityRatioStats(loss=float(loss.item()), acc=float(acc))

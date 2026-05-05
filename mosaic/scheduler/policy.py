"""
Scale Scheduler policy network π^sched_ψ(ℓ_t, b_t | φ_t).

Two-headed architecture:
  - Scale head: categorical distribution over {S, T, O}
  - Budget head: continuous scalar in [B_min, B_max]
Also includes a value head for PPO training.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from mosaic.scheduler.features import FeatureEncoder


class Scale(IntEnum):
    STRATEGIC = 0
    TACTICAL = 1
    OPERATIONAL = 2


class SchedulerAction(NamedTuple):
    scale: Scale
    budget: int
    scale_logprob: torch.Tensor
    budget_logprob: torch.Tensor   # log-prob under discretized Gaussian
    value: torch.Tensor


class ScaleSchedulerPolicy(nn.Module):
    """
    The Scale Scheduler: a lightweight network (≈1.2M params) that
    determines which scale to invoke and how many tokens to allocate.

    Architecture:
        raw features (5-d)
          → FeatureEncoder (2-layer MLP → 64-d)
          → shared trunk (64 → 256 → 256)
          → scale head (256 → 3)          [Categorical]
          → budget head (256 → 1)         [Sigmoid → [B_min, B_max]]
          → value head (256 → 1)          [PPO critic]
    """

    def __init__(
        self,
        raw_dim: int = 5,
        encoder_hidden: int = 256,
        encoder_out: int = 64,
        trunk_hidden: int = 256,
        num_scales: int = 3,
        budget_min: int = 256,
        budget_max: int = 6144,
    ):
        super().__init__()
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.num_scales = num_scales

        # Feature encoder
        self.encoder = FeatureEncoder(raw_dim, encoder_hidden, encoder_out)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(encoder_out, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

        # Heads
        self.scale_head = nn.Linear(trunk_hidden, num_scales)
        self.budget_head = nn.Linear(trunk_hidden, 1)
        self.value_head = nn.Linear(trunk_hidden, 1)

    def forward(self, raw_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_features: (batch, 5)
        Returns:
            scale_logits: (batch, 3)
            budget_frac:  (batch, 1) in [0, 1]
            value:        (batch, 1)
        """
        encoded = self.encoder(raw_features)
        trunk_out = self.trunk(encoded)

        scale_logits = self.scale_head(trunk_out)
        budget_frac = torch.sigmoid(self.budget_head(trunk_out))
        value = self.value_head(trunk_out)

        return scale_logits, budget_frac, value

    def act(self, raw_features: torch.Tensor, deterministic: bool = False) -> SchedulerAction:
        """
        Sample (or greedily select) an action given raw features.

        Args:
            raw_features: (5,) or (1, 5)
        Returns:
            SchedulerAction with scale, budget, log-probs, and value.
        """
        if raw_features.dim() == 1:
            raw_features = raw_features.unsqueeze(0)

        scale_logits, budget_frac, value = self.forward(raw_features)

        # Scale selection
        dist = Categorical(logits=scale_logits)
        if deterministic:
            scale_idx = scale_logits.argmax(dim=-1)
        else:
            scale_idx = dist.sample()
        scale_logprob = dist.log_prob(scale_idx)

        # Budget
        frac = budget_frac.squeeze(-1).item()
        budget = int(self.budget_min + (self.budget_max - self.budget_min) * frac)

        # Discretize budget logprob (treat sigmoid output as deterministic mapping)
        budget_logprob = torch.tensor(0.0)  # placeholder; exact in Gaussian version

        scale = Scale(scale_idx.item())

        return SchedulerAction(
            scale=scale,
            budget=budget,
            scale_logprob=scale_logprob.squeeze(),
            budget_logprob=budget_logprob,
            value=value.squeeze(),
        )

    def evaluate_actions(
        self,
        raw_features: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs and entropy for a batch of past actions.
        Used in PPO update.

        Args:
            raw_features: (batch, 5)
            scales: (batch,) LongTensor of scale indices
        Returns:
            scale_logprobs: (batch,)
            entropy: (batch,)
            values: (batch,)
        """
        scale_logits, budget_frac, values = self.forward(raw_features)
        dist = Categorical(logits=scale_logits)
        scale_logprobs = dist.log_prob(scales)
        entropy = dist.entropy()
        return scale_logprobs, entropy, values.squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

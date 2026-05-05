"""
PPO Trainer for the Scale Scheduler.

Trains the scheduler policy π^sched_ψ using Proximal Policy Optimization
with a composite reward r_t = r^task_t − λ · (b_t / B_max).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mosaic.config import PPOConfig
from mosaic.scheduler.policy import ScaleSchedulerPolicy, Scale

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single timestep's data for PPO."""
    features: torch.Tensor        # (5,)
    scale: int                    # index
    budget: int
    scale_logprob: float
    reward: float
    value: float
    done: bool


@dataclass
class RolloutBuffer:
    """Stores transitions for a batch of rollouts."""
    transitions: list[Transition] = field(default_factory=list)

    def append(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        n = len(self.transitions)
        rewards = torch.tensor([t.reward for t in self.transitions])
        values = torch.tensor([t.value for t in self.transitions])
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.float32)

        advantages = torch.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else 0.0
            next_done = dones[t + 1] if t + 1 < n else 1.0
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns, advantages

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "features": torch.stack([t.features for t in self.transitions]),
            "scales": torch.tensor([t.scale for t in self.transitions], dtype=torch.long),
            "budgets": torch.tensor([t.budget for t in self.transitions], dtype=torch.float32),
            "old_logprobs": torch.tensor([t.scale_logprob for t in self.transitions]),
        }


def compute_composite_reward(
    task_reward: float,
    budget_used: int,
    budget_max: int,
    lambda_efficiency: float,
) -> float:
    """r_t = r^task_t - λ · (b_t / B_max)"""
    return task_reward - lambda_efficiency * (budget_used / budget_max)


class PPOTrainer:
    """PPO trainer for the Scale Scheduler."""

    def __init__(
        self,
        policy: ScaleSchedulerPolicy,
        config: PPOConfig,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.buffer = RolloutBuffer()

    def store_transition(self, transition: Transition):
        self.buffer.append(transition)

    def update(self) -> dict[str, float]:
        """Run a single PPO update epoch on the buffer."""
        if len(self.buffer) == 0:
            return {}

        cfg = self.config

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = self.buffer.to_tensors()
        features = data["features"].to(self.device)
        old_scales = data["scales"].to(self.device)
        old_logprobs = data["old_logprobs"].to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # PPO update (multiple epochs over the same data)
        n = len(features)
        batch_size = min(cfg.batch_size, n)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _epoch in range(4):  # K epochs
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                b_features = features[idx]
                b_scales = old_scales[idx]
                b_old_logprobs = old_logprobs[idx]
                b_returns = returns[idx]
                b_advantages = advantages[idx]

                # Evaluate current policy
                new_logprobs, entropy, values = self.policy.evaluate_actions(
                    b_features, b_scales
                )

                # Policy loss (clipped PPO)
                ratio = (new_logprobs - b_old_logprobs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(
                    ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio
                ) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, b_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_coeff * value_loss
                    + cfg.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)
        logger.info(f"Scheduler saved to {path}")

    def load(self, path: str):
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        logger.info(f"Scheduler loaded from {path}")

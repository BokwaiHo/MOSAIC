"""
AgentConfig: central configuration for MOSAIC.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LayerConfig:
    """Per-layer (scale) configuration."""
    reasoning_budget: int = 2048          # default token budget B^ℓ
    context_window: int = 20             # sliding window size w
    temperature: float = 0.0
    max_retries: int = 2


@dataclass
class SchedulerConfig:
    """Scale Scheduler configuration."""
    hidden_dim: int = 256
    num_layers: int = 2
    feature_dim: int = 64                # dimension of φ_t after encoding
    budget_min: int = 256
    budget_max: int = 6144
    checkpoint_path: Optional[str] = None  # path to trained scheduler weights


@dataclass
class PPOConfig:
    """PPO training hyperparameters for the Scale Scheduler."""
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    batch_size: int = 64
    num_steps: int = 5000
    rollouts_per_step: int = 8
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class AgentConfig:
    """Top-level MOSAIC agent configuration."""

    # ----- Backbone LLM -----
    backbone_model: str = "Qwen/Qwen3.5-35B-A3B"
    backend: str = "transformers"        # "transformers" | "vllm"
    device: str = "auto"

    # ----- Layer budgets & windows -----
    strategic: LayerConfig = field(default_factory=lambda: LayerConfig(
        reasoning_budget=4096, context_window=2048, temperature=0.0,
    ))
    tactical: LayerConfig = field(default_factory=lambda: LayerConfig(
        reasoning_budget=2048, context_window=20, temperature=0.0,
    ))
    operational: LayerConfig = field(default_factory=lambda: LayerConfig(
        reasoning_budget=512, context_window=5, temperature=0.0,
    ))

    # ----- Scheduler -----
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # ----- PPO (used only during training) -----
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # ----- Efficiency penalty -----
    lambda_efficiency: float = 0.1

    # ----- Task limits -----
    max_steps: int = 500
    summary_max_tokens: int = 2048       # cap for trajectory summary

    # ----- I/O -----
    verbose: bool = False

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Recursively instantiate nested dataclasses
        for key in ("strategic", "tactical", "operational"):
            if key in raw:
                raw[key] = LayerConfig(**raw[key])
        if "scheduler" in raw:
            raw["scheduler"] = SchedulerConfig(**raw["scheduler"])
        if "ppo" in raw:
            raw["ppo"] = PPOConfig(**raw["ppo"])
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

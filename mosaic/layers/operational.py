"""
Operational Layer (π^O): single-step action execution.

Handles the majority of steps (~75%) with minimal context and budget,
designed for fast reactive execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from mosaic.config import LayerConfig
from mosaic.utils.llm_backend import LLMBackend

logger = logging.getLogger(__name__)

OPERATIONAL_PROMPT = """\
You are an agent executing a task step by step.

## Current Tactical Plan
{tactical_plan}

## Recent Steps
{recent_steps}

## Current Observation
{observation}

## Instructions
Choose the single best next action to make progress on the tactical plan.
Output ONLY the action in the format:
ACTION: <tool_name> <arguments>

Be precise and concise."""


@dataclass
class OperationalOutput:
    action_text: str
    tokens_used: int
    action_entropy: float          # entropy of the action distribution
    hidden_state: Optional[torch.Tensor]  # for scheduler feature extraction


class OperationalLayer:
    """Operational layer π^O."""

    def __init__(self, llm: LLMBackend, config: LayerConfig):
        self.llm = llm
        self.config = config

    def act(
        self,
        tactical_plan: str,
        recent_steps: list[str],
        observation: str,
        budget: int | None = None,
    ) -> OperationalOutput:
        """Generate a single action."""
        budget = budget or self.config.reasoning_budget

        # Build sliding-window context
        window = recent_steps[-self.config.context_window:]
        recent_text = "\n".join(
            f"  Step {i+1}: {s}" for i, s in enumerate(window)
        ) if window else "  (no prior steps)"

        prompt = OPERATIONAL_PROMPT.format(
            tactical_plan=tactical_plan,
            recent_steps=recent_text,
            observation=observation if observation else "(no observation yet)",
        )

        resp = self.llm.generate(
            prompt,
            max_new_tokens=budget,
            return_hidden=True,
            return_logprobs=True,
        )

        # Compute action entropy from logprobs
        entropy = self._compute_entropy(resp.logprobs)

        logger.debug(f"[Operational:act] action='{resp.text[:80]}…', "
                      f"tokens={resp.tokens_used}, entropy={entropy:.3f}")

        return OperationalOutput(
            action_text=resp.text.strip(),
            tokens_used=resp.tokens_used,
            action_entropy=entropy,
            hidden_state=resp.hidden_states,
        )

    @staticmethod
    def _compute_entropy(logprobs: Optional[torch.Tensor]) -> float:
        """Compute mean per-token entropy from log-probabilities."""
        if logprobs is None:
            return 0.0
        # logprobs shape: (num_tokens, vocab_size)
        probs = logprobs.exp()
        token_entropy = -(probs * logprobs).sum(dim=-1)  # (num_tokens,)
        return token_entropy.mean().item()

    @staticmethod
    def parse_action(raw_text: str) -> tuple[str, str]:
        """Extract (tool_name, arguments) from the raw action text."""
        text = raw_text.strip()
        # Look for "ACTION: tool_name arguments"
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                parts = line[len("ACTION:"):].strip().split(maxsplit=1)
                tool = parts[0] if parts else "noop"
                args = parts[1] if len(parts) > 1 else ""
                return tool, args
        # Fallback: treat entire text as action
        return "raw", text

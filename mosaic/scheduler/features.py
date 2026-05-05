"""
Feature extraction for the Scale Scheduler.

Extracts the compact feature vector φ_t = [e_progress, e_diverge,
e_entropy, e_error, e_subtask] from the current agent state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mosaic.utils.task_graph import TaskStateGraph

logger = logging.getLogger(__name__)


@dataclass
class SchedulerFeatures:
    """Raw features before MLP encoding."""
    progress: float           # e_progress ∈ [0, 1]
    plan_divergence: float    # e_diverge ∈ [0, 1]
    action_entropy: float     # e_entropy ∈ R+
    error_flag: float         # e_error ∈ {0, 1}
    subtask_boundary: float   # e_subtask ∈ [0, 1]

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.progress,
            self.plan_divergence,
            self.action_entropy,
            self.error_flag,
            self.subtask_boundary,
        ], dtype=torch.float32)


class FeatureExtractor:
    """
    Extracts scheduler features from the current agent state.
    """

    def __init__(self, divergence_threshold: float = 0.5):
        self.divergence_threshold = divergence_threshold
        self._step_count = 0
        self._steps_since_subtask_change = 0
        self._current_subtask_id: Optional[str] = None

    def extract(
        self,
        graph: TaskStateGraph,
        action_entropy: float,
        error_occurred: bool,
        planned_action: str = "",
        actual_observation: str = "",
        current_subtask_id: Optional[str] = None,
    ) -> SchedulerFeatures:
        """Extract the 5-dimensional feature vector φ_t."""
        self._step_count += 1

        # e_progress: fraction of subgoals completed
        progress = graph.progress

        # e_diverge: plan divergence (simplified; in full impl use embedding cosine)
        divergence = self._compute_divergence(planned_action, actual_observation)

        # e_entropy: directly from operational layer
        entropy = action_entropy

        # e_error: binary flag
        error = 1.0 if error_occurred else 0.0

        # e_subtask: proximity to subtask boundary
        if current_subtask_id != self._current_subtask_id:
            self._current_subtask_id = current_subtask_id
            self._steps_since_subtask_change = 0
        self._steps_since_subtask_change += 1
        # Heuristic: boundary proximity increases as we spend more steps
        # on the same subtask (approaching its expected duration)
        subtask_boundary = min(1.0, self._steps_since_subtask_change / 15.0)

        return SchedulerFeatures(
            progress=progress,
            plan_divergence=divergence,
            action_entropy=entropy,
            error_flag=error,
            subtask_boundary=subtask_boundary,
        )

    def _compute_divergence(self, planned: str, observed: str) -> float:
        """
        Compute plan divergence between expected and actual outcome.

        In the full implementation, this uses embedding cosine distance.
        Here we use a simple heuristic based on keyword overlap.
        """
        if not planned or not observed:
            return 0.0

        planned_words = set(planned.lower().split())
        observed_words = set(observed.lower().split())

        if not planned_words:
            return 0.0

        overlap = len(planned_words & observed_words)
        similarity = overlap / max(len(planned_words), 1)

        return 1.0 - similarity  # divergence = 1 - similarity

    def reset(self):
        self._step_count = 0
        self._steps_since_subtask_change = 0
        self._current_subtask_id = None


class FeatureEncoder(nn.Module):
    """
    2-layer MLP that encodes raw features into the scheduler's
    input representation.
    """

    def __init__(self, raw_dim: int = 5, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, raw_dim) -> (batch, output_dim)"""
        return self.net(x)

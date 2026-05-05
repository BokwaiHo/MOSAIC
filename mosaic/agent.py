"""
MOSAICAgent: main inference loop implementing Algorithm 1 from the paper.

  for t = 0, 1, 2, ...
      φ_t ← ExtractFeatures(...)
      ℓ_t, b_t ← π^sched(φ_t)
      if ℓ_t = Strategic: distill, update graph, replan, refine
      elif ℓ_t = Tactical: distill, replan subtask
      a_t ← π^O(...)  [always executed]
      o_t ← Env(a_t)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import torch

from mosaic.config import AgentConfig
from mosaic.utils.llm_backend import LLMBackend
from mosaic.utils.task_graph import TaskStateGraph
from mosaic.layers.strategic import StrategicLayer
from mosaic.layers.tactical import TacticalLayer
from mosaic.layers.operational import OperationalLayer
from mosaic.distillation.distill_up import DistillUp
from mosaic.distillation.refine_down import RefineDown
from mosaic.scheduler.features import FeatureExtractor
from mosaic.scheduler.policy import ScaleSchedulerPolicy, Scale

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Environment interface
# ------------------------------------------------------------------
class Environment(Protocol):
    """Minimal environment interface."""

    def step(self, action: str) -> tuple[str, float, bool]:
        """
        Execute an action.
        Returns: (observation, reward, done)
        """
        ...

    def get_tool_inventory(self) -> str:
        """Return available tools as a string."""
        ...


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
@dataclass
class AgentResult:
    """Result of a single MOSAIC run."""
    final_output: str
    total_tokens: int
    total_steps: int
    trajectory: list[dict] = field(default_factory=list)
    scale_history: list[str] = field(default_factory=list)
    budget_history: list[int] = field(default_factory=list)

    @property
    def strategic_ratio(self) -> float:
        return self.scale_history.count("strategic") / max(len(self.scale_history), 1)

    @property
    def tactical_ratio(self) -> float:
        return self.scale_history.count("tactical") / max(len(self.scale_history), 1)

    @property
    def operational_ratio(self) -> float:
        return self.scale_history.count("operational") / max(len(self.scale_history), 1)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
class MOSAICAgent:
    """
    MOSAIC Agent: Multi-Scale Orchestrated Scale-Adaptive Inference Control.

    Implements Algorithm 1 from the paper.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # Backbone LLM
        self.llm = LLMBackend(
            model_name=config.backbone_model,
            backend=config.backend,
            device=config.device,
        )

        # Three-layer decision stack
        self.strategic = StrategicLayer(
            self.llm, config.strategic, config.summary_max_tokens,
        )
        self.tactical = TacticalLayer(self.llm, config.tactical)
        self.operational = OperationalLayer(self.llm, config.operational)

        # Inter-scale context distillation
        self.distill_up = DistillUp(self.llm)
        self.refine_down = RefineDown(self.llm)

        # Scale Scheduler
        self.feature_extractor = FeatureExtractor()
        self.scheduler = ScaleSchedulerPolicy(
            raw_dim=5,
            encoder_hidden=config.scheduler.hidden_dim,
            encoder_out=config.scheduler.feature_dim,
            trunk_hidden=config.scheduler.hidden_dim,
            budget_min=config.scheduler.budget_min,
            budget_max=config.scheduler.budget_max,
        )

        # Load trained scheduler weights if provided
        if config.scheduler.checkpoint_path:
            self.scheduler.load_state_dict(
                torch.load(config.scheduler.checkpoint_path,
                           map_location="cpu", weights_only=True)
            )
        self.scheduler.eval()

    # ------------------------------------------------------------------
    # Main inference loop (Algorithm 1)
    # ------------------------------------------------------------------
    def run(self, task_description: str, env: Optional[Environment] = None) -> AgentResult:
        """
        Execute the full MOSAIC inference loop.

        Args:
            task_description: natural language task description.
            env: environment that accepts actions and returns observations.
                 If None, runs in "plan-only" mode (no env interaction).
        Returns:
            AgentResult with trajectory, token counts, and scale history.
        """
        # ---- Initialization (Alg 1, lines 1-3) ----
        strategic_out = self.strategic.initialize(task_description)
        graph = strategic_out.graph
        strategic_plan = strategic_out.plan_text
        total_tokens = strategic_out.tokens_used

        # Refine into initial tactical plan
        current_subtask = graph.next_pending_subgoal()
        subtask_desc = current_subtask.description if current_subtask else task_description
        refine_out = self.refine_down.strategic_to_tactical(
            strategic_plan, subtask_desc,
            tool_inventory=env.get_tool_inventory() if env else "bash, file_edit",
        )
        tactical_plan = refine_out.plan_text
        total_tokens += refine_out.tokens_used

        # State tracking
        self.feature_extractor.reset()
        trajectory: list[dict] = []
        step_texts: list[str] = []
        scale_history: list[str] = []
        budget_history: list[int] = []
        subtask_log_start = 0
        observation = ""
        last_action = ""

        logger.info(f"[MOSAIC] Starting task with {len(graph.nodes)} subgoals")

        # ---- Main loop (Alg 1, lines 4-19) ----
        for t in range(self.config.max_steps):
            # Line 5: Extract scheduler features
            features = self.feature_extractor.extract(
                graph=graph,
                action_entropy=0.0,     # will be updated after operational act
                error_occurred="error" in observation.lower() if observation else False,
                planned_action=last_action,
                actual_observation=observation,
                current_subtask_id=current_subtask.id if current_subtask else None,
            )

            # Line 6: Select scale and budget
            with torch.no_grad():
                action = self.scheduler.act(
                    features.to_tensor(), deterministic=True
                )
            scale = action.scale
            budget = action.budget

            scale_name = Scale(scale).name.lower()
            scale_history.append(scale_name)
            budget_history.append(budget)

            if self.config.verbose:
                logger.info(f"  [Step {t}] scale={scale_name}, budget={budget}, "
                             f"progress={features.progress:.1%}")

            # Lines 7-14: Conditional scale invocation
            if scale == Scale.STRATEGIC:
                # Lines 8-11: Strategic replan
                subtask_log = "\n".join(step_texts[subtask_log_start:])
                if current_subtask:
                    dist_out = self.distill_up.tactical_to_strategic(
                        subtask_log,
                        subtask_name=current_subtask.description,
                        subtask_node_id=current_subtask.id,
                    )
                    graph.apply_distill_update(dist_out.structured_update)
                    total_tokens += dist_out.tokens_used

                strat_out = self.strategic.replan(
                    graph, "\n".join(step_texts[-50:]), budget=budget,
                )
                strategic_plan = strat_out.plan_text
                graph = strat_out.graph
                total_tokens += strat_out.tokens_used

                # Update subtask and refine
                current_subtask = graph.next_pending_subgoal()
                if current_subtask:
                    ref_out = self.refine_down.strategic_to_tactical(
                        strategic_plan, current_subtask.description,
                        tool_inventory=env.get_tool_inventory() if env else "bash, file_edit",
                        recent_context="\n".join(step_texts[-5:]),
                    )
                    tactical_plan = ref_out.plan_text
                    total_tokens += ref_out.tokens_used
                    subtask_log_start = len(step_texts)

            elif scale == Scale.TACTICAL:
                # Lines 13-14: Tactical replan
                recent_log = "\n".join(step_texts[-self.config.tactical.context_window:])
                dist_out = self.distill_up.operational_to_tactical(recent_log)
                total_tokens += dist_out.tokens_used

                tac_out = self.tactical.plan(
                    subtask=subtask_desc,
                    strategic_plan=strategic_plan,
                    execution_log=recent_log,
                    tool_inventory=env.get_tool_inventory() if env else "bash, file_edit",
                    budget=budget,
                )
                tactical_plan = tac_out.plan_text
                total_tokens += tac_out.tokens_used

            # Line 16: Execute operational action (always)
            op_budget = min(budget, self.config.operational.reasoning_budget)
            op_out = self.operational.act(
                tactical_plan=tactical_plan,
                recent_steps=step_texts[-self.config.operational.context_window:],
                observation=observation,
                budget=op_budget,
            )
            total_tokens += op_out.tokens_used

            # Update entropy for next step's feature extraction
            # (retroactively; the scheduler used the previous step's entropy)

            # Line 17: Environment interaction
            last_action = op_out.action_text
            if env is not None:
                tool, args = OperationalLayer.parse_action(op_out.action_text)
                observation, reward, done = env.step(f"{tool} {args}")
            else:
                observation = "(no environment)"
                reward = 0.0
                done = False

            # Record trajectory
            step_record = {
                "step": t,
                "scale": scale_name,
                "budget": budget,
                "action": op_out.action_text,
                "observation": observation[:500],
                "tokens": op_out.tokens_used,
            }
            trajectory.append(step_record)
            step_texts.append(f"Action: {op_out.action_text} | Obs: {observation[:200]}")

            # Line 18: Check termination
            if done:
                logger.info(f"[MOSAIC] Task completed at step {t}, "
                             f"total_tokens={total_tokens}")
                break

        # ---- Build result ----
        return AgentResult(
            final_output=observation,
            total_tokens=total_tokens,
            total_steps=len(trajectory),
            trajectory=trajectory,
            scale_history=scale_history,
            budget_history=budget_history,
        )

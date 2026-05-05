"""
Strategic Layer (π^S): task-level goal decomposition and global replanning.

Operates over the entire task horizon.  Invoked at initialization and at
escalation points detected by the Scale Scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from mosaic.config import LayerConfig
from mosaic.utils.llm_backend import LLMBackend, LLMResponse
from mosaic.utils.task_graph import TaskStateGraph, GRAPH_INIT_PROMPT

logger = logging.getLogger(__name__)

STRATEGIC_PLAN_PROMPT = """\
You are a strategic planner for an autonomous agent.

## Current Task State
{graph_str}

## Trajectory Summary
{summary}

## Instructions
Based on the current state, produce a revised strategic plan:
1. Identify which subgoals are complete, in progress, or blocked.
2. Re-prioritize remaining subgoals if needed.
3. Output an ordered list of next subtask specifications.

Respond concisely (<{budget} tokens). Focus on WHAT to do, not HOW."""

SUMMARIZE_PROMPT = """\
Summarize the following agent trajectory into a concise overview (<{max_tokens} tokens).
Focus on: actions taken, outcomes, errors encountered, and current state.

Trajectory:
{trajectory}"""


@dataclass
class StrategicOutput:
    plan_text: str
    graph: TaskStateGraph
    tokens_used: int


class StrategicLayer:
    """Strategic layer π^S."""

    def __init__(self, llm: LLMBackend, config: LayerConfig, summary_max_tokens: int = 2048):
        self.llm = llm
        self.config = config
        self.summary_max_tokens = summary_max_tokens

    def initialize(self, task_description: str) -> StrategicOutput:
        """Build the initial task state graph and strategic plan."""
        # Step 1: build graph
        prompt = GRAPH_INIT_PROMPT.format(task_description=task_description)
        resp = self.llm.generate(prompt, max_new_tokens=self.config.reasoning_budget)
        graph = TaskStateGraph.from_llm_output(resp.text)

        # Step 2: generate initial plan
        plan_prompt = STRATEGIC_PLAN_PROMPT.format(
            graph_str=graph.to_prompt_str(),
            summary="Task just started. No actions taken yet.",
            budget=self.config.reasoning_budget,
        )
        plan_resp = self.llm.generate(
            plan_prompt, max_new_tokens=self.config.reasoning_budget,
        )

        total_tokens = resp.tokens_used + plan_resp.tokens_used
        logger.info(f"[Strategic:init] graph nodes={len(graph.nodes)}, "
                     f"tokens={total_tokens}")

        return StrategicOutput(
            plan_text=plan_resp.text,
            graph=graph,
            tokens_used=total_tokens,
        )

    def replan(
        self,
        graph: TaskStateGraph,
        trajectory_text: str,
        budget: int | None = None,
    ) -> StrategicOutput:
        """Re-invoke strategic reasoning to revise the plan."""
        budget = budget or self.config.reasoning_budget

        # Compress trajectory
        summary = self._summarize_trajectory(trajectory_text)

        prompt = STRATEGIC_PLAN_PROMPT.format(
            graph_str=graph.to_prompt_str(),
            summary=summary,
            budget=budget,
        )
        resp = self.llm.generate(prompt, max_new_tokens=budget)

        logger.info(f"[Strategic:replan] progress={graph.progress:.1%}, "
                     f"tokens={resp.tokens_used}")

        return StrategicOutput(
            plan_text=resp.text,
            graph=graph,
            tokens_used=resp.tokens_used,
        )

    def _summarize_trajectory(self, trajectory_text: str) -> str:
        if self.llm.count_tokens(trajectory_text) <= self.summary_max_tokens:
            return trajectory_text
        prompt = SUMMARIZE_PROMPT.format(
            trajectory=trajectory_text,
            max_tokens=self.summary_max_tokens,
        )
        resp = self.llm.generate(prompt, max_new_tokens=self.summary_max_tokens)
        return resp.text

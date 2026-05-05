"""
Tactical Layer (π^T): subtask-level planning with tool selection.

Operates at the subtask level (5–20 steps).  Bridges strategic goals
and operational actions by producing step-by-step plans with checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mosaic.config import LayerConfig
from mosaic.utils.llm_backend import LLMBackend

logger = logging.getLogger(__name__)

TACTICAL_PLAN_PROMPT = """\
You are a tactical planner for an autonomous agent.

## Current Subtask
{subtask}

## Strategic Plan Context
{strategic_plan}

## Recent Execution Log (last {window} steps)
{execution_log}

## Available Tools
{tool_inventory}

## Instructions
Generate a tactical plan for the current subtask:
1. List 5–10 concrete steps with specific tool/action selections.
2. Define intermediate checkpoints to verify progress.
3. Include fallback actions if a step fails.

Be concrete and actionable. Respond concisely (<{budget} tokens)."""

TACTICAL_REPLAN_PROMPT = """\
You are a tactical planner. Your previous plan encountered an issue.

## Original Tactical Plan
{original_plan}

## Issue Encountered
{issue}

## Recent Execution Log
{execution_log}

## Instructions
Revise the tactical plan to address the issue.
Keep unchanged steps and modify only what is necessary.
Respond concisely (<{budget} tokens)."""


@dataclass
class TacticalOutput:
    plan_text: str
    tokens_used: int


class TacticalLayer:
    """Tactical layer π^T."""

    def __init__(self, llm: LLMBackend, config: LayerConfig):
        self.llm = llm
        self.config = config

    def plan(
        self,
        subtask: str,
        strategic_plan: str,
        execution_log: str,
        tool_inventory: str = "bash, file_edit, web_search, code_execute",
        budget: int | None = None,
    ) -> TacticalOutput:
        """Generate a new tactical plan for the given subtask."""
        budget = budget or self.config.reasoning_budget

        prompt = TACTICAL_PLAN_PROMPT.format(
            subtask=subtask,
            strategic_plan=strategic_plan,
            execution_log=execution_log,
            window=self.config.context_window,
            tool_inventory=tool_inventory,
            budget=budget,
        )
        resp = self.llm.generate(prompt, max_new_tokens=budget)

        logger.info(f"[Tactical:plan] subtask='{subtask[:60]}…', "
                     f"tokens={resp.tokens_used}")

        return TacticalOutput(plan_text=resp.text, tokens_used=resp.tokens_used)

    def replan(
        self,
        original_plan: str,
        issue: str,
        execution_log: str,
        budget: int | None = None,
    ) -> TacticalOutput:
        """Revise an existing tactical plan after encountering an issue."""
        budget = budget or self.config.reasoning_budget

        prompt = TACTICAL_REPLAN_PROMPT.format(
            original_plan=original_plan,
            issue=issue,
            execution_log=execution_log,
            budget=budget,
        )
        resp = self.llm.generate(prompt, max_new_tokens=budget)

        logger.info(f"[Tactical:replan] issue='{issue[:60]}…', "
                     f"tokens={resp.tokens_used}")

        return TacticalOutput(plan_text=resp.text, tokens_used=resp.tokens_used)

"""
Refine↓ operator: expand high-level directives into actionable
instructions for lower layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mosaic.utils.llm_backend import LLMBackend

logger = logging.getLogger(__name__)

REFINE_STR_TO_TAC_PROMPT = """\
You are a tactical planner. Given the strategic plan and current context, \
generate a detailed tactical plan for the next subtask.

Strategic plan: {strategic_plan}
Current subtask: {current_subtask}
Available tools: {tool_inventory}
Recent context: {recent_context}

Output a step-by-step plan with:
1. Specific tool/action selections for each step
2. Expected intermediate checkpoints
3. Fallback actions if a step fails
Keep the plan to 5–10 steps. Respond concisely."""


@dataclass
class RefineOutput:
    plan_text: str
    tokens_used: int


class RefineDown:
    """Refine↓ operator for expanding directives into actionable plans."""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def strategic_to_tactical(
        self,
        strategic_plan: str,
        current_subtask: str,
        tool_inventory: str = "bash, file_edit, web_search, code_execute",
        recent_context: str = "",
        max_tokens: int = 1024,
    ) -> RefineOutput:
        """Expand a strategic subtask into a tactical plan."""
        prompt = REFINE_STR_TO_TAC_PROMPT.format(
            strategic_plan=strategic_plan,
            current_subtask=current_subtask,
            tool_inventory=tool_inventory,
            recent_context=recent_context or "(none)",
        )
        resp = self.llm.generate(prompt, max_new_tokens=max_tokens)

        logger.debug(f"[Refine↓:Str→Tac] subtask='{current_subtask[:50]}…', "
                      f"tokens={resp.tokens_used}")

        return RefineOutput(plan_text=resp.text, tokens_used=resp.tokens_used)

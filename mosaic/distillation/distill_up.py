"""
Distill↑ operator: compress lower-layer execution logs into structured
summaries for higher layers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from mosaic.utils.llm_backend import LLMBackend

logger = logging.getLogger(__name__)

DISTILL_OP_TO_TAC_PROMPT = """\
You are summarizing recent execution steps for a tactical planner.
Given the operational log below, produce a brief structured summary (<150 tokens) with:
- **Actions taken**: List of actions executed
- **Results**: What was observed or achieved
- **Blockers**: Any errors or unexpected outcomes

Operational log:
{operational_log}

Respond with ONLY the structured summary."""

DISTILL_TAC_TO_STR_PROMPT = """\
You are reporting subtask results to a strategic planner.
Given the tactical execution log for the subtask "{subtask_name}", produce a structured JSON report with:
- "node_id": the subtask identifier
- "outcome": "success" | "partial" | "failure"
- "key_findings": list of important discoveries (strings)
- "unexpected_events": list of deviations from the plan (strings)

Tactical log:
{tactical_log}

Respond with ONLY the JSON object, no other text."""


@dataclass
class DistillOutput:
    summary_text: str
    structured_update: dict     # parsed JSON for graph update
    tokens_used: int


class DistillUp:
    """Distill↑ operator for compressing lower-layer logs."""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def operational_to_tactical(self, operational_log: str) -> DistillOutput:
        """Compress operational logs into a tactical-level summary."""
        prompt = DISTILL_OP_TO_TAC_PROMPT.format(operational_log=operational_log)
        resp = self.llm.generate(prompt, max_new_tokens=200)

        logger.debug(f"[Distill↑:Op→Tac] tokens={resp.tokens_used}")
        return DistillOutput(
            summary_text=resp.text,
            structured_update={},
            tokens_used=resp.tokens_used,
        )

    def tactical_to_strategic(
        self, tactical_log: str, subtask_name: str, subtask_node_id: str,
    ) -> DistillOutput:
        """Compress tactical logs into a strategic-level structured report."""
        prompt = DISTILL_TAC_TO_STR_PROMPT.format(
            tactical_log=tactical_log,
            subtask_name=subtask_name,
        )
        resp = self.llm.generate(prompt, max_new_tokens=300)

        # Parse structured update
        update = self._parse_update(resp.text, subtask_node_id)

        logger.debug(f"[Distill↑:Tac→Str] outcome={update.get('outcome','?')}, "
                      f"tokens={resp.tokens_used}")
        return DistillOutput(
            summary_text=resp.text,
            structured_update=update,
            tokens_used=resp.tokens_used,
        )

    @staticmethod
    def _parse_update(text: str, fallback_node_id: str) -> dict:
        """Parse LLM output into a structured update dict."""
        try:
            # Try direct JSON parse
            data = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        # Ensure required fields
        data.setdefault("node_id", fallback_node_id)
        data.setdefault("outcome", "partial")
        data.setdefault("key_findings", [])
        data.setdefault("unexpected_events", [])
        return data

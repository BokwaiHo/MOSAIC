"""
Task State Graph G_t = (V_t, E_t).

Maintains the structured representation of goals, subgoals, constraints,
and their dependencies / completion status for the Strategic layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class NodeType(str, Enum):
    GOAL = "goal"
    SUBGOAL = "subgoal"
    CONSTRAINT = "constraint"


class EdgeRelation(str, Enum):
    DEPENDS_ON = "depends_on"
    BLOCKS = "blocks"


@dataclass
class GraphNode:
    id: str
    description: str
    node_type: NodeType = NodeType.SUBGOAL
    status: NodeStatus = NodeStatus.PENDING


@dataclass
class GraphEdge:
    source: str        # node id
    target: str        # node id
    relation: EdgeRelation = EdgeRelation.DEPENDS_ON


@dataclass
class TaskStateGraph:
    """
    Task state graph G_t that the Strategic layer uses as context.
    """
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_llm_output(cls, json_str: str) -> "TaskStateGraph":
        """Parse an LLM-generated JSON into a TaskStateGraph."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_str)
            if match:
                data = json.loads(match.group(1))
            else:
                logger.warning("Failed to parse graph JSON; creating empty graph.")
                return cls()

        graph = cls()
        for node_data in data.get("nodes", []):
            node = GraphNode(
                id=node_data["id"],
                description=node_data["description"],
                node_type=NodeType(node_data.get("type", "subgoal")),
                status=NodeStatus(node_data.get("status", "pending")),
            )
            graph.nodes[node.id] = node

        for edge_data in data.get("edges", []):
            edge = GraphEdge(
                source=edge_data["from"],
                target=edge_data["to"],
                relation=EdgeRelation(edge_data.get("relation", "depends_on")),
            )
            graph.edges.append(edge)

        return graph

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_node_status(self, node_id: str, status: NodeStatus):
        if node_id in self.nodes:
            self.nodes[node_id].status = status

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def add_constraint(self, description: str) -> str:
        cid = f"constraint_{len(self.nodes)}"
        self.add_node(GraphNode(
            id=cid, description=description,
            node_type=NodeType.CONSTRAINT, status=NodeStatus.PENDING,
        ))
        return cid

    def apply_distill_update(self, update: dict):
        """
        Apply a Distill↑ structured update.
        Expected keys: node_id, outcome, key_findings, unexpected_events.
        """
        node_id = update.get("node_id")
        outcome = update.get("outcome", "").lower()

        if node_id and node_id in self.nodes:
            if outcome in ("success", "done"):
                self.update_node_status(node_id, NodeStatus.DONE)
            elif outcome in ("failure", "failed"):
                self.update_node_status(node_id, NodeStatus.FAILED)
            elif outcome in ("partial",):
                self.update_node_status(node_id, NodeStatus.IN_PROGRESS)

        # Add newly discovered constraints
        for event in update.get("unexpected_events", []):
            if event:
                self.add_constraint(event)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def progress(self) -> float:
        """Fraction of non-constraint nodes completed."""
        actionable = [n for n in self.nodes.values()
                      if n.node_type != NodeType.CONSTRAINT]
        if not actionable:
            return 0.0
        done = sum(1 for n in actionable if n.status == NodeStatus.DONE)
        return done / len(actionable)

    def next_pending_subgoal(self) -> Optional[GraphNode]:
        """Return the first pending subgoal whose dependencies are met."""
        completed = {n.id for n in self.nodes.values()
                     if n.status == NodeStatus.DONE}
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            if node.node_type == NodeType.CONSTRAINT:
                continue
            # Check all dependencies satisfied
            deps = [e.source for e in self.edges
                    if e.target == node.id and e.relation == EdgeRelation.DEPENDS_ON]
            if all(d in completed for d in deps):
                return node
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_prompt_str(self) -> str:
        """Serialize the graph into a compact string for LLM context."""
        lines = ["## Task State Graph"]
        for node in self.nodes.values():
            status_icon = {
                NodeStatus.PENDING: "○",
                NodeStatus.IN_PROGRESS: "◑",
                NodeStatus.DONE: "●",
                NodeStatus.FAILED: "✗",
            }[node.status]
            lines.append(
                f"  {status_icon} [{node.id}] ({node.node_type.value}) "
                f"{node.description}"
            )
        if self.edges:
            lines.append("  Dependencies:")
            for e in self.edges:
                lines.append(f"    {e.source} --{e.relation.value}--> {e.target}")
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "nodes": [
                {"id": n.id, "description": n.description,
                 "type": n.node_type.value, "status": n.status.value}
                for n in self.nodes.values()
            ],
            "edges": [
                {"from": e.source, "to": e.target, "relation": e.relation.value}
                for e in self.edges
            ],
        }
        return json.dumps(data, indent=2)


# ------------------------------------------------------------------
# Graph initialization prompt
# ------------------------------------------------------------------
GRAPH_INIT_PROMPT = """\
Analyze the following task and decompose it into a dependency graph of subgoals.

Task: {task_description}

Output a JSON object with:
- "nodes": list of {{"id": str, "description": str, "type": "goal"|"subgoal"|"constraint", "status": "pending"}}
- "edges": list of {{"from": str, "to": str, "relation": "depends_on"|"blocks"}}

Respond with ONLY the JSON, no other text."""

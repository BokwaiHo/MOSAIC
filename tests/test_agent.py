"""
Unit tests for MOSAIC components.

Run with: pytest tests/test_agent.py -v
"""

import json
import pytest
import torch
import numpy as np

from mosaic.config import AgentConfig, LayerConfig, SchedulerConfig, PPOConfig
from mosaic.utils.task_graph import (
    TaskStateGraph, GraphNode, GraphEdge,
    NodeType, NodeStatus, EdgeRelation,
)
from mosaic.scheduler.features import FeatureExtractor, FeatureEncoder, SchedulerFeatures
from mosaic.scheduler.policy import ScaleSchedulerPolicy, Scale, SchedulerAction
from mosaic.scheduler.trainer import (
    PPOTrainer, Transition, RolloutBuffer, compute_composite_reward,
)


# ==================================================================
# Config
# ==================================================================
class TestConfig:
    def test_default_config(self):
        cfg = AgentConfig()
        assert cfg.strategic.reasoning_budget == 4096
        assert cfg.tactical.reasoning_budget == 2048
        assert cfg.operational.reasoning_budget == 512
        assert cfg.lambda_efficiency == 0.1
        assert cfg.max_steps == 500

    def test_yaml_roundtrip(self, tmp_path):
        cfg = AgentConfig(backbone_model="test-model", lambda_efficiency=0.2)
        yaml_path = tmp_path / "test.yaml"
        cfg.to_yaml(yaml_path)
        loaded = AgentConfig.from_yaml(yaml_path)
        assert loaded.backbone_model == "test-model"
        assert loaded.lambda_efficiency == 0.2


# ==================================================================
# Task State Graph
# ==================================================================
class TestTaskStateGraph:
    def test_empty_graph(self):
        g = TaskStateGraph()
        assert len(g.nodes) == 0
        assert g.progress == 0.0
        assert g.next_pending_subgoal() is None

    def test_add_nodes_and_progress(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "subtask A", NodeType.SUBGOAL, NodeStatus.PENDING))
        g.add_node(GraphNode("b", "subtask B", NodeType.SUBGOAL, NodeStatus.DONE))
        assert g.progress == 0.5

    def test_dependency_ordering(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "first", NodeType.SUBGOAL, NodeStatus.PENDING))
        g.add_node(GraphNode("b", "second", NodeType.SUBGOAL, NodeStatus.PENDING))
        g.edges.append(GraphEdge("a", "b", EdgeRelation.DEPENDS_ON))

        # "b" depends on "a", so next pending should be "a"
        nxt = g.next_pending_subgoal()
        assert nxt is not None
        assert nxt.id == "a"

        # Complete "a", now "b" should be next
        g.update_node_status("a", NodeStatus.DONE)
        nxt = g.next_pending_subgoal()
        assert nxt is not None
        assert nxt.id == "b"

    def test_from_llm_output(self):
        llm_json = json.dumps({
            "nodes": [
                {"id": "g1", "description": "fix bug", "type": "goal", "status": "pending"},
                {"id": "s1", "description": "find file", "type": "subgoal", "status": "pending"},
            ],
            "edges": [
                {"from": "s1", "to": "g1", "relation": "depends_on"},
            ],
        })
        g = TaskStateGraph.from_llm_output(llm_json)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_from_llm_output_with_markdown(self):
        llm_out = '```json\n{"nodes": [], "edges": []}\n```'
        g = TaskStateGraph.from_llm_output(llm_out)
        assert len(g.nodes) == 0

    def test_apply_distill_update(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("s1", "subtask 1", NodeType.SUBGOAL, NodeStatus.PENDING))

        update = {
            "node_id": "s1",
            "outcome": "success",
            "key_findings": ["found the bug"],
            "unexpected_events": ["different module affected"],
        }
        g.apply_distill_update(update)

        assert g.nodes["s1"].status == NodeStatus.DONE
        # Should have added a constraint node for unexpected event
        assert len(g.nodes) == 2

    def test_to_prompt_str(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "task A", NodeType.SUBGOAL, NodeStatus.DONE))
        s = g.to_prompt_str()
        assert "●" in s  # done icon
        assert "task A" in s

    def test_add_constraint(self):
        g = TaskStateGraph()
        cid = g.add_constraint("do not modify API")
        assert cid in g.nodes
        assert g.nodes[cid].node_type == NodeType.CONSTRAINT
        # Constraints should not affect progress
        assert g.progress == 0.0


# ==================================================================
# Feature Extractor
# ==================================================================
class TestFeatureExtractor:
    def test_basic_extraction(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "task A", NodeType.SUBGOAL, NodeStatus.DONE))
        g.add_node(GraphNode("b", "task B", NodeType.SUBGOAL, NodeStatus.PENDING))

        fe = FeatureExtractor()
        feats = fe.extract(
            graph=g,
            action_entropy=1.5,
            error_occurred=True,
            planned_action="search for bug",
            actual_observation="file not found error",
        )

        assert feats.progress == 0.5
        assert feats.action_entropy == 1.5
        assert feats.error_flag == 1.0
        assert 0.0 <= feats.plan_divergence <= 1.0

    def test_to_tensor(self):
        feats = SchedulerFeatures(
            progress=0.5,
            plan_divergence=0.3,
            action_entropy=1.2,
            error_flag=0.0,
            subtask_boundary=0.7,
        )
        t = feats.to_tensor()
        assert t.shape == (5,)
        assert t[0].item() == pytest.approx(0.5)

    def test_subtask_boundary_increases(self):
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "task", NodeType.SUBGOAL, NodeStatus.PENDING))

        fe = FeatureExtractor()
        boundaries = []
        for _ in range(20):
            f = fe.extract(g, 0.0, False, current_subtask_id="a")
            boundaries.append(f.subtask_boundary)

        # Should be monotonically increasing (up to 1.0)
        for i in range(1, len(boundaries)):
            assert boundaries[i] >= boundaries[i - 1]


# ==================================================================
# Feature Encoder (MLP)
# ==================================================================
class TestFeatureEncoder:
    def test_forward_shape(self):
        enc = FeatureEncoder(raw_dim=5, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 5)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_single_sample(self):
        enc = FeatureEncoder()
        x = torch.randn(1, 5)
        out = enc(x)
        assert out.shape == (1, 64)  # default output_dim


# ==================================================================
# Scale Scheduler Policy
# ==================================================================
class TestSchedulerPolicy:
    def setup_method(self):
        self.policy = ScaleSchedulerPolicy(
            raw_dim=5, encoder_hidden=64, encoder_out=32,
            trunk_hidden=64, budget_min=256, budget_max=4096,
        )

    def test_parameter_count(self):
        # Should be in the ballpark of ~100K for small config
        assert self.policy.num_parameters > 1000
        assert self.policy.num_parameters < 10_000_000

    def test_forward_shapes(self):
        x = torch.randn(8, 5)
        logits, frac, value = self.policy(x)
        assert logits.shape == (8, 3)
        assert frac.shape == (8, 1)
        assert value.shape == (8, 1)
        # Budget fraction should be in [0, 1]
        assert (frac >= 0).all() and (frac <= 1).all()

    def test_act_deterministic(self):
        phi = torch.randn(5)
        a1 = self.policy.act(phi, deterministic=True)
        a2 = self.policy.act(phi, deterministic=True)
        assert a1.scale == a2.scale
        assert a1.budget == a2.budget

    def test_act_returns_valid_scale(self):
        phi = torch.randn(5)
        action = self.policy.act(phi)
        assert action.scale in (Scale.STRATEGIC, Scale.TACTICAL, Scale.OPERATIONAL)
        assert 256 <= action.budget <= 4096

    def test_evaluate_actions(self):
        features = torch.randn(16, 5)
        scales = torch.randint(0, 3, (16,))
        logprobs, entropy, values = self.policy.evaluate_actions(features, scales)
        assert logprobs.shape == (16,)
        assert entropy.shape == (16,)
        assert values.shape == (16,)
        assert (entropy >= 0).all()


# ==================================================================
# PPO Trainer
# ==================================================================
class TestPPOTrainer:
    def test_composite_reward(self):
        r = compute_composite_reward(
            task_reward=1.0, budget_used=2048, budget_max=4096,
            lambda_efficiency=0.1,
        )
        assert r == pytest.approx(1.0 - 0.1 * 0.5)

    def test_rollout_buffer(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.append(Transition(
                features=torch.randn(5),
                scale=i % 3,
                budget=512,
                scale_logprob=-1.0,
                reward=0.1 * i,
                value=0.05 * i,
                done=(i == 9),
            ))
        assert len(buf) == 10

        returns, advantages = buf.compute_returns_and_advantages(0.99, 0.95)
        assert returns.shape == (10,)
        assert advantages.shape == (10,)

    def test_single_update(self):
        policy = ScaleSchedulerPolicy(
            raw_dim=5, encoder_hidden=32, encoder_out=16,
            trunk_hidden=32,
        )
        trainer = PPOTrainer(policy, PPOConfig(batch_size=4), device="cpu")

        # Add some transitions
        for i in range(20):
            trainer.store_transition(Transition(
                features=torch.randn(5),
                scale=np.random.randint(3),
                budget=512,
                scale_logprob=-1.0,
                reward=np.random.randn() * 0.1,
                value=np.random.randn() * 0.1,
                done=(i == 19),
            ))

        metrics = trainer.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert len(trainer.buffer) == 0  # buffer cleared after update

    def test_save_load(self, tmp_path):
        policy = ScaleSchedulerPolicy(raw_dim=5, encoder_hidden=32,
                                       encoder_out=16, trunk_hidden=32)
        trainer = PPOTrainer(policy, PPOConfig(), device="cpu")
        path = str(tmp_path / "test.pt")
        trainer.save(path)
        trainer.load(path)


# ==================================================================
# Integration: end-to-end scheduler step
# ==================================================================
class TestIntegration:
    def test_feature_to_action_pipeline(self):
        """Test the full pipeline: features → encoder → policy → action."""
        g = TaskStateGraph()
        g.add_node(GraphNode("a", "task A", NodeType.SUBGOAL, NodeStatus.PENDING))

        fe = FeatureExtractor()
        feats = fe.extract(g, action_entropy=0.8, error_occurred=False)

        policy = ScaleSchedulerPolicy(
            raw_dim=5, encoder_hidden=64, encoder_out=32, trunk_hidden=64,
        )

        action = policy.act(feats.to_tensor(), deterministic=True)
        assert isinstance(action, SchedulerAction)
        assert isinstance(action.scale, Scale)
        assert isinstance(action.budget, int)

    def test_graph_lifecycle(self):
        """Test graph init → update → progress tracking."""
        g = TaskStateGraph()
        for i in range(5):
            g.add_node(GraphNode(
                f"s{i}", f"subtask {i}",
                NodeType.SUBGOAL, NodeStatus.PENDING,
            ))

        assert g.progress == 0.0

        # Complete subtasks one by one
        for i in range(5):
            g.update_node_status(f"s{i}", NodeStatus.DONE)
            expected = (i + 1) / 5
            assert g.progress == pytest.approx(expected)

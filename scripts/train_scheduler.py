#!/usr/bin/env python3
"""
Train the MOSAIC Scale Scheduler via PPO.

Usage:
    python scripts/train_scheduler.py \
        --config configs/default.yaml \
        --train_tasks configs/train_tasks.json \
        --output_dir checkpoints/scheduler \
        --lambda_efficiency 0.1 \
        --ppo_steps 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import numpy as np

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
from mosaic.scheduler.trainer import (
    PPOTrainer, Transition, RolloutBuffer, compute_composite_reward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_scheduler")


# ------------------------------------------------------------------
# Simulated training environment
# ------------------------------------------------------------------
class SimulatedTrainingEnv:
    """
    Lightweight simulated environment for scheduler training.

    In a full implementation, this would wrap the actual benchmark
    environments (SWE-Bench, BrowseComp, WebArena, GAIA).
    Here we provide a simplified simulation that generates plausible
    rewards based on task progress and error patterns.
    """

    def __init__(self, task: dict, max_steps: int = 100):
        self.task = task
        self.max_steps = max_steps
        self.step_count = 0
        self.description = task.get("description", "Simulated task")
        self.difficulty = task.get("difficulty", 0.5)  # [0, 1]
        self.num_subtasks = task.get("num_subtasks", 5)
        self._rng = np.random.RandomState(hash(self.description) % 2**31)

        # Simulate subtask boundaries and error-prone steps
        self._error_steps = set(
            self._rng.choice(max_steps, size=max(1, int(max_steps * 0.1)), replace=False)
        )
        self._subtask_boundaries = set(
            np.linspace(0, max_steps, self.num_subtasks + 1, dtype=int)[1:-1]
        )

    def reset(self) -> str:
        self.step_count = 0
        return f"Task: {self.description}"

    def step(self, action: str) -> tuple[str, float, bool]:
        self.step_count += 1

        # Simulate observation
        error = self.step_count in self._error_steps
        at_boundary = self.step_count in self._subtask_boundaries
        progress = self.step_count / self.max_steps

        if error:
            obs = f"[ERROR] Tool execution failed at step {self.step_count}"
            reward = -0.1
        elif at_boundary:
            obs = f"[CHECKPOINT] Subtask completed at step {self.step_count}"
            reward = 0.3
        else:
            obs = f"Step {self.step_count}: Action executed. Progress: {progress:.0%}"
            reward = 0.01 + 0.02 * (1 - self.difficulty)

        done = self.step_count >= self.max_steps or (
            progress > 0.9 and self._rng.random() < 0.3
        )
        if done and progress > 0.7:
            reward += 1.0  # task completion bonus

        return obs, reward, done

    def get_tool_inventory(self) -> str:
        return "bash, file_edit, web_search, code_execute"


def load_train_tasks(path: str) -> list[dict]:
    """Load training tasks from JSON or generate defaults."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        logger.warning(f"Train tasks file not found at {path}. "
                       f"Generating synthetic tasks.")
        tasks = []
        for i in range(200):
            benchmark = ["swebench", "browsecomp", "webarena", "gaia"][i % 4]
            tasks.append({
                "id": f"{benchmark}_{i // 4}",
                "description": f"Simulated {benchmark} task #{i // 4}",
                "benchmark": benchmark,
                "difficulty": np.random.uniform(0.3, 0.9),
                "num_subtasks": np.random.randint(3, 8),
            })
        return tasks


# ------------------------------------------------------------------
# Rollout collection
# ------------------------------------------------------------------
def collect_rollout(
    policy: ScaleSchedulerPolicy,
    env: SimulatedTrainingEnv,
    config: AgentConfig,
    feature_extractor: FeatureExtractor,
) -> list[Transition]:
    """Run one episode and collect transitions for PPO."""
    feature_extractor.reset()
    obs = env.reset()
    transitions = []

    # Simplified graph for training
    graph = TaskStateGraph()
    for i in range(env.num_subtasks):
        from mosaic.utils.task_graph import GraphNode, NodeType, NodeStatus
        graph.add_node(GraphNode(
            id=f"sub_{i}", description=f"Subtask {i}",
            node_type=NodeType.SUBGOAL, status=NodeStatus.PENDING,
        ))

    for t in range(env.max_steps):
        error_occurred = "ERROR" in obs
        at_boundary = "CHECKPOINT" in obs

        # Update graph on checkpoints
        if at_boundary:
            for node in graph.nodes.values():
                if node.status == NodeStatus.PENDING:
                    node.status = NodeStatus.DONE
                    break

        features = feature_extractor.extract(
            graph=graph,
            action_entropy=np.random.exponential(0.5),  # simulated
            error_occurred=error_occurred,
            planned_action="expected_action",
            actual_observation=obs,
            current_subtask_id=graph.next_pending_subgoal().id
            if graph.next_pending_subgoal() else None,
        )

        phi = features.to_tensor()
        with torch.no_grad():
            action = policy.act(phi, deterministic=False)

        # Execute in environment
        obs, task_reward, done = env.step(f"action_{t}")

        # Composite reward
        reward = compute_composite_reward(
            task_reward=task_reward,
            budget_used=action.budget,
            budget_max=config.scheduler.budget_max,
            lambda_efficiency=config.lambda_efficiency,
        )

        transitions.append(Transition(
            features=phi,
            scale=action.scale.value,
            budget=action.budget,
            scale_logprob=action.scale_logprob.item(),
            reward=reward,
            value=action.value.item(),
            done=done,
        ))

        if done:
            break

    return transitions


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MOSAIC Scale Scheduler")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train_tasks", type=str, default="configs/train_tasks.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/scheduler")
    parser.add_argument("--lambda_efficiency", type=float, default=None)
    parser.add_argument("--ppo_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Config
    if os.path.exists(args.config):
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()

    if args.lambda_efficiency is not None:
        config.lambda_efficiency = args.lambda_efficiency
    if args.ppo_steps is not None:
        config.ppo.num_steps = args.ppo_steps

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize policy
    policy = ScaleSchedulerPolicy(
        budget_min=config.scheduler.budget_min,
        budget_max=config.scheduler.budget_max,
    )
    logger.info(f"Scheduler parameters: {policy.num_parameters:,}")

    # Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(policy, config.ppo, device=device)

    # Tasks
    tasks = load_train_tasks(args.train_tasks)
    logger.info(f"Loaded {len(tasks)} training tasks")

    # Training loop
    best_reward = float("-inf")
    feature_extractor = FeatureExtractor()

    logger.info(f"Starting training for {config.ppo.num_steps} PPO steps")
    start_time = time.time()

    for step in range(config.ppo.num_steps):
        # Collect rollouts
        step_rewards = []
        for _ in range(config.ppo.rollouts_per_step):
            task = tasks[np.random.randint(len(tasks))]
            env = SimulatedTrainingEnv(task, max_steps=100)
            transitions = collect_rollout(
                policy, env, config, feature_extractor,
            )
            for t in transitions:
                trainer.store_transition(t)
            ep_reward = sum(t.reward for t in transitions)
            step_rewards.append(ep_reward)

        # PPO update
        metrics = trainer.update()
        mean_reward = np.mean(step_rewards)

        # Logging
        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Step {step + 1}/{config.ppo.num_steps} | "
                f"reward={mean_reward:.3f} | "
                f"policy_loss={metrics.get('policy_loss', 0):.4f} | "
                f"value_loss={metrics.get('value_loss', 0):.4f} | "
                f"entropy={metrics.get('entropy', 0):.4f} | "
                f"elapsed={elapsed:.0f}s"
            )

        # Save best
        if mean_reward > best_reward:
            best_reward = mean_reward
            trainer.save(str(output_dir / "best.pt"))

        # Periodic checkpoint
        if (step + 1) % 500 == 0:
            trainer.save(str(output_dir / f"step_{step + 1}.pt"))

    # Final save
    trainer.save(str(output_dir / "final.pt"))
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time / 3600:.1f} hours. "
                f"Best reward: {best_reward:.3f}")


if __name__ == "__main__":
    main()

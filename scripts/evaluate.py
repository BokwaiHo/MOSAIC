#!/usr/bin/env python3
"""
Evaluate MOSAIC on supported benchmarks.

Usage:
    python scripts/evaluate.py \
        --benchmark swebench \
        --backbone Qwen/Qwen3.5-35B-A3B \
        --scheduler_ckpt checkpoints/scheduler/best.pt \
        --lambda_efficiency 0.1 \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

from mosaic.config import AgentConfig
from mosaic.agent import MOSAICAgent, AgentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


# ------------------------------------------------------------------
# Benchmark environment adapters
# ------------------------------------------------------------------
class BenchmarkEnv:
    """
    Base class for benchmark environment adapters.

    Each subclass wraps a real benchmark (SWE-Bench, BrowseComp, etc.)
    into the Environment protocol expected by MOSAICAgent.
    """

    def __init__(self, task: dict):
        self.task = task
        self.task_id = task.get("id", "unknown")
        self.steps = 0
        self.max_steps = task.get("max_steps", 200)
        self.history: list[dict] = []

    def step(self, action: str) -> tuple[str, float, bool]:
        """Execute action and return (observation, reward, done)."""
        self.steps += 1
        observation = self._execute(action)
        reward = self._compute_reward(observation)
        done = self._check_done(observation)
        self.history.append({
            "step": self.steps,
            "action": action,
            "observation": observation[:500],
            "reward": reward,
        })
        return observation, reward, done

    def get_tool_inventory(self) -> str:
        return "bash, file_edit, web_search, code_execute"

    def _execute(self, action: str) -> str:
        raise NotImplementedError

    def _compute_reward(self, observation: str) -> float:
        return 0.0

    def _check_done(self, observation: str) -> bool:
        return self.steps >= self.max_steps


class SWEBenchEnv(BenchmarkEnv):
    """
    Adapter for SWE-Bench Verified.

    In a full implementation, this would:
    1. Spin up a Docker container with the repo at the correct commit.
    2. Execute bash commands submitted by the agent.
    3. Run the test suite and compute pass rate deltas.
    """

    def _execute(self, action: str) -> str:
        # Placeholder: in production, execute in Docker container
        return f"[SWE-Bench] Executed: {action[:100]}... (simulated)"

    def _compute_reward(self, observation: str) -> float:
        # Placeholder: in production, run tests and compute delta
        return 0.01

    def _check_done(self, observation: str) -> bool:
        if self.steps >= self.max_steps:
            return True
        if "All tests passed" in observation:
            return True
        return False

    def get_tool_inventory(self) -> str:
        return "bash, file_read, file_edit, grep, find, python"


class BrowseCompEnv(BenchmarkEnv):
    """Adapter for BrowseComp."""

    def _execute(self, action: str) -> str:
        return f"[BrowseComp] Search result for: {action[:100]}... (simulated)"

    def _compute_reward(self, observation: str) -> float:
        return 0.0  # sparse: only on final answer

    def get_tool_inventory(self) -> str:
        return "web_search, web_fetch, extract_text"


class WebArenaEnv(BenchmarkEnv):
    """Adapter for WebArena."""

    def _execute(self, action: str) -> str:
        return f"[WebArena] Page after action: {action[:100]}... (simulated)"

    def _compute_reward(self, observation: str) -> float:
        return 0.01

    def get_tool_inventory(self) -> str:
        return "click, type, scroll, navigate, go_back"


class GAIAEnv(BenchmarkEnv):
    """Adapter for GAIA Level 2&3."""

    def _execute(self, action: str) -> str:
        return f"[GAIA] Result: {action[:100]}... (simulated)"

    def _compute_reward(self, observation: str) -> float:
        return 0.0  # sparse: only on final answer

    def get_tool_inventory(self) -> str:
        return "web_search, code_execute, file_read, calculator"


BENCHMARK_ENVS = {
    "swebench": SWEBenchEnv,
    "browsecomp": BrowseCompEnv,
    "webarena": WebArenaEnv,
    "gaia": GAIAEnv,
}


# ------------------------------------------------------------------
# Task loading
# ------------------------------------------------------------------
def load_benchmark_tasks(benchmark: str, data_dir: str = "data") -> list[dict]:
    """
    Load tasks for a benchmark.

    In production, this loads from the actual benchmark datasets.
    Here we generate placeholder tasks for demonstration.
    """
    task_file = Path(data_dir) / f"{benchmark}_tasks.json"
    if task_file.exists():
        with open(task_file) as f:
            return json.load(f)

    logger.warning(f"Task file {task_file} not found. Using placeholder tasks.")
    tasks = []
    num_tasks = {"swebench": 500, "browsecomp": 100, "webarena": 100, "gaia": 50}
    n = num_tasks.get(benchmark, 50)
    for i in range(n):
        tasks.append({
            "id": f"{benchmark}_{i:04d}",
            "description": f"Placeholder {benchmark} task #{i}",
            "max_steps": 200,
        })
    return tasks


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
def evaluate_single(
    agent: MOSAICAgent,
    task: dict,
    env_cls: type,
) -> dict:
    """Evaluate on a single task."""
    env = env_cls(task)
    start = time.time()

    try:
        result = agent.run(task["description"], env=env)
    except Exception as e:
        logger.error(f"Task {task['id']} failed: {e}")
        result = AgentResult(
            final_output=str(e),
            total_tokens=0,
            total_steps=0,
        )

    elapsed = time.time() - start

    return {
        "task_id": task["id"],
        "total_tokens": result.total_tokens,
        "total_steps": result.total_steps,
        "time_seconds": round(elapsed, 2),
        "strategic_ratio": round(result.strategic_ratio, 4),
        "tactical_ratio": round(result.tactical_ratio, 4),
        "operational_ratio": round(result.operational_ratio, 4),
        "final_output": result.final_output[:500],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOSAIC")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_ENVS.keys()))
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--scheduler_ckpt", type=str, default=None)
    parser.add_argument("--lambda_efficiency", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit number of tasks (for debugging)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Config
    if os.path.exists(args.config):
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()

    config.backbone_model = args.backbone
    if args.scheduler_ckpt:
        config.scheduler.checkpoint_path = args.scheduler_ckpt
    if args.lambda_efficiency is not None:
        config.lambda_efficiency = args.lambda_efficiency

    # Agent
    logger.info(f"Initializing MOSAIC with backbone={config.backbone_model}")
    agent = MOSAICAgent(config)

    # Tasks
    tasks = load_benchmark_tasks(args.benchmark, args.data_dir)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    logger.info(f"Evaluating on {len(tasks)} {args.benchmark} tasks")

    env_cls = BENCHMARK_ENVS[args.benchmark]

    # Run
    all_results = []
    for i, task in enumerate(tasks):
        logger.info(f"[{i + 1}/{len(tasks)}] Task: {task['id']}")
        result = evaluate_single(agent, task, env_cls)
        all_results.append(result)

        if (i + 1) % 10 == 0:
            avg_tokens = np.mean([r["total_tokens"] for r in all_results])
            avg_steps = np.mean([r["total_steps"] for r in all_results])
            logger.info(f"  Progress: {i + 1}/{len(tasks)} | "
                        f"avg_tokens={avg_tokens:.0f} | avg_steps={avg_steps:.1f}")

    # Aggregate metrics
    summary = {
        "benchmark": args.benchmark,
        "backbone": args.backbone,
        "lambda_efficiency": config.lambda_efficiency,
        "num_tasks": len(all_results),
        "avg_tokens": round(np.mean([r["total_tokens"] for r in all_results]), 1),
        "std_tokens": round(np.std([r["total_tokens"] for r in all_results]), 1),
        "avg_steps": round(np.mean([r["total_steps"] for r in all_results]), 1),
        "avg_time_sec": round(np.mean([r["time_seconds"] for r in all_results]), 2),
        "avg_strategic_ratio": round(np.mean([r["strategic_ratio"] for r in all_results]), 4),
        "avg_tactical_ratio": round(np.mean([r["tactical_ratio"] for r in all_results]), 4),
        "avg_operational_ratio": round(np.mean([r["operational_ratio"] for r in all_results]), 4),
    }

    # Save
    output_dir = Path(args.output_dir) / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION SUMMARY: {args.benchmark}")
    logger.info(f"{'=' * 60}")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

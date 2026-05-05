# MOSAIC: Scaling Long-Horizon Language Agents via Multi-Scale Adaptive Inference Control

Official implementation for the paper *"MOSAIC: Scaling Long-Horizon Language Agents via Multi-Scale Adaptive Inference Control"*.

## Overview

MOSAIC is a hierarchical agent framework that decomposes long-horizon decision making into three temporal scales: **Strategic**, **Tactical**, and **Operational**, each governed by distinct context scopes and reasoning budgets. A learned **Scale Scheduler** dynamically routes decisions to the appropriate scale with an adaptive token budget.


## Installation

```bash
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.1, vLLM ≥ 0.6 (optional, for fast inference).

## Quick Start

```python
from mosaic import MOSAICAgent, AgentConfig

config = AgentConfig(
    backbone_model="Qwen/Qwen3.5-35B-A3B",
    lambda_efficiency=0.1,
)
agent = MOSAICAgent(config)
result = agent.run("Fix the bug in utils.py that causes test_parse to fail.")
print(result.final_output)
print(f"Total tokens used: {result.total_tokens}")
```

## Training the Scale Scheduler

```bash
python scripts/train_scheduler.py \
    --backbone Qwen/Qwen3.5-35B-A3B \
    --train_tasks configs/train_tasks.json \
    --lambda_efficiency 0.1 \
    --ppo_steps 5000 \
    --output_dir checkpoints/scheduler
```

## Running Evaluation

```bash
# SWE-Bench Verified
python scripts/evaluate.py --benchmark swebench --backbone Qwen/Qwen3.5-35B-A3B

# BrowseComp
python scripts/evaluate.py --benchmark browsecomp --backbone Qwen/Qwen3.5-35B-A3B

# WebArena
python scripts/evaluate.py --benchmark webarena --backbone Qwen/Qwen3.5-35B-A3B

# GAIA Level 2&3
python scripts/evaluate.py --benchmark gaia --backbone Qwen/Qwen3.5-35B-A3B
```

## Project Structure

```
MOSAIC/
├── mosaic/
│   ├── __init__.py              # Public API
│   ├── agent.py                 # MOSAICAgent: main inference loop (Algorithm 1)
│   ├── config.py                # AgentConfig dataclass
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── strategic.py         # Strategic layer (π^S)
│   │   ├── tactical.py          # Tactical layer (π^T)
│   │   └── operational.py       # Operational layer (π^O)
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── features.py          # Feature extraction (φ_t)
│   │   ├── policy.py            # Scale Scheduler policy network
│   │   └── trainer.py           # PPO trainer for the scheduler
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── distill_up.py        # Distill↑ operator
│   │   └── refine_down.py       # Refine↓ operator
│   └── utils/
│       ├── __init__.py
│       ├── llm_backend.py       # LLM inference wrapper
│       └── task_graph.py        # Task state graph G_t
├── configs/
│   └── default.yaml             # Default configuration
├── scripts/
│   ├── train_scheduler.py       # Scheduler training script
│   └── evaluate.py              # Evaluation script
├── tests/
│   └── test_agent.py            # Unit tests
├── setup.py
└── README.md
```

## License

MIT License

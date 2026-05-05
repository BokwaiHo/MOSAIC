"""
MOSAIC: Multi-Scale Orchestrated Scale-Adaptive Inference Control.

A hierarchical agent framework for scaling long-horizon language agents.
"""

__version__ = "0.1.0"

from mosaic.config import AgentConfig, LayerConfig, SchedulerConfig, PPOConfig
from mosaic.agent import MOSAICAgent, AgentResult
from mosaic.scheduler.policy import Scale

__all__ = [
    "MOSAICAgent",
    "AgentConfig",
    "AgentResult",
    "LayerConfig",
    "SchedulerConfig",
    "PPOConfig",
    "Scale",
]

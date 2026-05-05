from setuptools import setup, find_packages

setup(
    name="mosaic-agent",
    version="0.1.0",
    description="MOSAIC: Scaling Long-Horizon Language Agents via Multi-Scale Adaptive Inference Control",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.6"],
        "dev": ["pytest>=7.0", "black", "ruff"],
    },
)

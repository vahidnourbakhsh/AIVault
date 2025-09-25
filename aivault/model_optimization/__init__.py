"""
Model Optimization module containing techniques for model compression and optimization.

This module includes:
- Model quantization
- Model pruning
- Knowledge distillation
"""

from . import quantization
from . import pruning
from . import distillation

__all__ = [
    "quantization",
    "pruning",
    "distillation"
]

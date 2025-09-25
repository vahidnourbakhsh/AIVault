"""
Machine Learning module containing classical ML algorithms and implementations.

This module includes:
- Supervised learning algorithms
- Unsupervised learning methods
- Reinforcement learning implementations
"""

from . import supervised_learning
from . import unsupervised_learning  
from . import reinforcement_learning

__all__ = [
    "supervised_learning",
    "unsupervised_learning",
    "reinforcement_learning"
]

"""
AIVault: A comprehensive collection of AI models, methods, examples, and tutorials.

This package provides practical implementations and educational resources for various
AI techniques including Generative AI, Machine Learning, Deep Learning, Computer Vision,
Natural Language Processing, and more.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import utilities
from . import generative_ai
from . import machine_learning
from . import deep_learning
from . import computer_vision
from . import nlp
from . import model_optimization
from . import deployment

__all__ = [
    "utilities",
    "generative_ai", 
    "machine_learning",
    "deep_learning",
    "computer_vision",
    "nlp",
    "model_optimization",
    "deployment"
]

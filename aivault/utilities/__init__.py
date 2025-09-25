"""
Utilities module containing helper functions and common utilities.

This module includes:
- Data loading and preprocessing utilities
- Visualization helpers
- Model evaluation metrics
- Common helper functions
"""

# Import modules to make them available
from . import data_utils
from . import viz_utils
from . import eval_utils
from . import common_utils

# Import key functions directly for convenience
try:
    from .data_utils import load_dataset, preprocess_data
    from .viz_utils import plot_training_curves, plot_confusion_matrix
    from .eval_utils import calculate_metrics
    from .common_utils import setup_logging, set_seed
    
    __all__ = [
        "data_utils",
        "viz_utils", 
        "eval_utils",
        "common_utils",
        "load_dataset",
        "preprocess_data",
        "plot_training_curves",
        "plot_confusion_matrix", 
        "calculate_metrics",
        "setup_logging",
        "set_seed"
    ]
except ImportError:
    # If dependencies are not available, only expose modules
    __all__ = [
        "data_utils",
        "viz_utils", 
        "eval_utils",
        "common_utils"
    ]

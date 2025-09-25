"""
Utilities module containing helper functions and common utilities.

This module includes:
- Data loading and preprocessing utilities
- Visualization helpers
- Model evaluation metrics
- Common helper functions
"""

# Import modules to make them available
try:
    from . import data_utils
    from . import viz_utils
    from . import eval_utils
    from . import common_utils
    
    # Import key functions directly for convenience
    from .data_utils import load_dataset, preprocess_data, get_data_info
    from .viz_utils import plot_training_curves, plot_confusion_matrix, set_plot_style
    from .eval_utils import calculate_metrics
    from .common_utils import setup_logging, set_seed
    
    __all__ = [
        "data_utils",
        "viz_utils", 
        "eval_utils",
        "common_utils",
        "load_dataset",
        "preprocess_data",
        "get_data_info",
        "plot_training_curves",
        "plot_confusion_matrix",
        "set_plot_style", 
        "calculate_metrics",
        "setup_logging",
        "set_seed"
    ]
except ImportError:
    # If dependencies are not available, only expose basic modules
    try:
        from . import common_utils
        __all__ = ["common_utils"]
    except ImportError:
        __all__ = []

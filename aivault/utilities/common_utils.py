"""
Common utilities and helper functions.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for deep learning frameworks if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
        
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon GPU
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def count_parameters(model: Any) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model object
        
    Returns:
        Number of trainable parameters
    """
    try:
        # For PyTorch models
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError:
        try:
            # For TensorFlow/Keras models
            return model.count_params()
        except AttributeError:
            return 0


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> None:
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        filepath: Local file path to save
        chunk_size: Chunk size for downloading
    """
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=Path(filepath).name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_bytes(psutil.virtual_memory().total),
        "memory_available": format_bytes(psutil.virtual_memory().available),
    }
    
    # GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = format_bytes(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        pass
    
    return info

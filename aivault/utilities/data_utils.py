"""
Data utilities for loading, preprocessing, and handling datasets.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def load_dataset(
    data_path: Union[str, Path], 
    file_type: str = "auto"
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset from various file formats.
    
    Args:
        data_path: Path to the data file
        file_type: Type of file ('csv', 'json', 'pickle', 'auto')
        
    Returns:
        Loaded data as DataFrame or dictionary
    """
    data_path = Path(data_path)
    
    if file_type == "auto":
        file_type = data_path.suffix.lower()[1:]
    
    if file_type == "csv":
        return pd.read_csv(data_path)
    elif file_type == "json":
        with open(data_path, 'r') as f:
            return json.load(f)
    elif file_type in ["pkl", "pickle"]:
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for machine learning.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
        scale_features: Whether to scale features
        scaler_type: Type of scaler ('standard' or 'minmax')
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    if scale_features:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def save_data(data: Any, filepath: Union[str, Path], file_type: str = "pickle") -> None:
    """
    Save data to file.
    
    Args:
        data: Data to save
        filepath: Output file path
        file_type: File format ('pickle', 'json', 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if file_type == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif file_type == "json":
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif file_type == "csv" and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file type or data format: {file_type}")


def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "memory_usage": data.memory_usage(deep=True).sum(),
        "numerical_columns": list(data.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(data.select_dtypes(include=['object']).columns),
        "unique_counts": {col: data[col].nunique() for col in data.columns}
    }
    
    return info

"""
Evaluation utilities for model assessment and metrics calculation.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification",
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('classification' or 'regression')
        average: Averaging strategy for multiclass ('weighted', 'macro', 'micro')
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    if task_type == "classification":
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average)
        
        # ROC AUC for binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            
    elif task_type == "regression":
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))


def calculate_model_performance(
    models: Dict[str, any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "classification"
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary of model performances
    """
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, task_type)
        results[model_name] = metrics
        
    return results


def cross_validation_scores(
    model: any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Union[str, List[str]] = "accuracy"
) -> Dict[str, List[float]]:
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Targets
        cv: Number of cross-validation folds
        scoring: Scoring metric(s)
        
    Returns:
        Dictionary of cross-validation scores
    """
    from sklearn.model_selection import cross_val_score
    
    if isinstance(scoring, str):
        scoring = [scoring]
    
    results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        results[metric] = scores.tolist()
        
    return results


def calculate_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> tuple:
    """
    Calculate confidence interval for scores.
    
    Args:
        scores: List of scores
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    scores = np.array(scores)
    mean_score = np.mean(scores)
    sem = stats.sem(scores)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
    
    return mean_score - h, mean_score + h


def model_summary(
    model: any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "classification"
) -> Dict[str, any]:
    """
    Generate comprehensive model summary.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        task_type: Type of task
        
    Returns:
        Dictionary with model summary
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, task_type)
    test_metrics = calculate_metrics(y_test, y_test_pred, task_type)
    
    summary = {
        "model_name": type(model).__name__,
        "task_type": task_type,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X_train.shape[1],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    return summary

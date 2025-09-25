"""
Visualization utilities for plotting and displaying results.
"""

from typing import List, Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values  
        train_accuracies: Training accuracy values
        val_accuracies: Validation accuracy values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies if provided
    if train_accuracies and val_accuracies:
        axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm))
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importances: Feature importance values
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_data_distribution(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    plot_type: str = 'hist',
    save_path: Optional[str] = None
) -> None:
    """
    Plot data distribution for numerical columns.
    
    Args:
        data: Input DataFrame
        columns: Columns to plot (default: all numerical)
        plot_type: Type of plot ('hist', 'box', 'kde')
        save_path: Path to save the plot
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, col in enumerate(columns):
        if plot_type == 'hist':
            data[col].hist(ax=axes[i], bins=30)
        elif plot_type == 'box':
            data.boxplot(column=col, ax=axes[i])
        elif plot_type == 'kde':
            data[col].plot.kde(ax=axes[i])
        
        axes[i].set_title(f'{col} Distribution')
        axes[i].grid(True)
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def set_plot_style(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Set matplotlib plotting style.
    
    Args:
        style: Matplotlib style name
        figsize: Default figure size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12

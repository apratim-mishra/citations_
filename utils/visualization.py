"""
Visualization utilities for plotting results.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_metrics(
    metrics: Dict[str, float],
    k_values: List[int],
    save_path: Optional[str] = None,
    title: str = "Evaluation Metrics",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot multiple metrics at different k values.
    
    Args:
        metrics: Dictionary of metric names to values
        k_values: List of k values for which metrics are available
        save_path: Path to save the plot (if None, plot is not saved)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter and group metrics
    k_metrics = {}
    other_metrics = {}
    
    for metric, value in metrics.items():
        if any(f"@{k}" in metric for k in k_values):
            # This is a k-based metric
            base_metric = metric.split('@')[0]
            k = int(metric.split('@')[1])
            
            if base_metric not in k_metrics:
                k_metrics[base_metric] = {}
            
            k_metrics[base_metric][k] = value
        else:
            # This is a non-k metric
            other_metrics[metric] = value
    
    # Plot k-based metrics
    for metric_name, k_values_dict in k_metrics.items():
        k_list = sorted(k_values_dict.keys())
        values = [k_values_dict[k] for k in k_list]
        ax.plot(k_list, values, 'o-', linewidth=2, label=metric_name)
    
    # Add labels and title
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add other metrics as text
    if other_metrics:
        text = "\n".join([f"{metric}: {value:.4f}" for metric, value in other_metrics.items()])
        plt.figtext(0.02, 0.02, text, fontsize=10)
    
    # Customize grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Set x-axis to only show the k values used
    ax.set_xticks(sorted(k_list))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auc_score: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
        auc_score: Area under the PR curve
        save_path: Path to save the plot (if None, plot is not saved)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, 'b-', linewidth=2)
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.2, color='b')
    
    # Add labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    
    if auc_score is not None:
        title = f"{title} (AUC: {auc_score:.4f})"
    
    ax.set_title(title, fontsize=14)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid and baseline
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=np.mean(precision), color='r', linestyle='--', alpha=0.5, label='Baseline')
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_model_comparison(
    all_metrics: Dict[str, Dict[str, float]],
    k_values: List[int],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        all_metrics: Dictionary of model names to their metric dictionaries
        k_values: List of k values for which metrics are available
        metrics_to_plot: List of metric names to plot (if None, plots all available)
        save_path: Path to save the plot (if None, plot is not saved)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Determine which metrics to plot
    if metrics_to_plot is None:
        # Find common metrics across all models
        common_metrics = set()
        for model_name, metrics in all_metrics.items():
            if common_metrics:
                common_metrics &= set(metrics.keys())
            else:
                common_metrics = set(metrics.keys())
        
        # Filter to only k-based metrics
        k_based_metrics = set()
        for metric in common_metrics:
            if any(f"@{k}" in metric for k in k_values):
                base_metric = metric.split('@')[0]
                k_based_metrics.add(base_metric)
        
        metrics_to_plot = sorted(list(k_based_metrics))
    
    # Create subplots for each metric
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize, sharey=False)
    
    # Handle case with only one metric
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Plot for each model
        for model_name, metrics in all_metrics.items():
            # Extract values for this metric across k values
            k_vals = []
            metric_vals = []
            
            for k in k_values:
                key = f"{metric_name}@{k}"
                if key in metrics:
                    k_vals.append(k)
                    metric_vals.append(metrics[key])
            
            if k_vals:
                ax.plot(k_vals, metric_vals, 'o-', linewidth=2, label=model_name)
        
        # Customize subplot
        ax.set_title(f"{metric_name.upper()}", fontsize=12)
        ax.set_xlabel('k', fontsize=10)
        
        # Set y-label only for leftmost subplot
        if i == 0:
            ax.set_ylabel('Metric Value', fontsize=10)
        
        # Set x-axis ticks
        ax.set_xticks(k_values)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend to the right of the last subplot
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=10)
    
    # Add overall title
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_node_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    n_components: int = 2,
    save_path: Optional[str] = None,
    title: str = "Node Embeddings",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot node embeddings in 2D or 3D.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels for coloring (optional)
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_components: Number of components (2 or 3)
        save_path: Path to save the plot (if None, plot is not saved)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3")
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create figure
    if n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        # Plot with labels
        if n_components == 3:
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                reduced_embeddings[:, 2],
                c=labels,
                cmap='viridis',
                alpha=0.7
            )
        else:
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.7
            )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        # Plot without labels
        if n_components == 3:
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                reduced_embeddings[:, 2],
                alpha=0.7
            )
        else:
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.7
            )
    
    # Add labels and title
    if n_components == 3:
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.set_zlabel('Component 3', fontsize=10)
    else:
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
    
    ax.set_title(f"{title} ({method.upper()})", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary of metrics to their values over epochs
        metrics: List of metrics to plot (if None, plots all available)
        save_path: Path to save the plot (if None, plot is not saved)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Determine which metrics to plot
    if metrics is None:
        metrics = list(history.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each metric
    for metric in metrics:
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'o-', linewidth=2, label=metric)
    
    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Customize grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
"""
Classification metrics for evaluating binary link prediction.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score, classification_report as sklearn_classification_report

def auc_score(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate ROC AUC score for binary link prediction.
    
    Args:
        predictions: Predicted scores (higher = more likely to be positive)
        targets: Ground truth binary labels
        
    Returns:
        ROC AUC score
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate AUC score
    return roc_auc_score(targets, predictions)

def precision_recall_auc(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Precision-Recall AUC and return PR curve data.
    
    Args:
        predictions: Predicted scores
        targets: Ground truth binary labels
        
    Returns:
        Tuple of (PR AUC score, precision array, recall array, thresholds array)
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(targets, predictions)
    
    # Calculate AUC of precision-recall curve
    pr_auc = auc(recall, precision)
    
    return pr_auc, precision, recall, thresholds

def f1_score(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    average: str = 'binary'
) -> float:
    """
    Calculate F1 score for binary link prediction.
    
    Args:
        predictions: Predicted scores
        targets: Ground truth binary labels
        threshold: Threshold for converting scores to binary predictions
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        F1 score
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert scores to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Calculate F1 score
    return sklearn_f1_score(targets, binary_preds, average=average)

def accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> float:
    """
    Calculate accuracy for binary link prediction.
    
    Args:
        predictions: Predicted scores
        targets: Ground truth binary labels
        threshold: Threshold for converting scores to binary predictions
        
    Returns:
        Accuracy score
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert scores to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Calculate accuracy
    return accuracy_score(targets, binary_preds)

def classification_report(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    output_dict: bool = True
) -> Union[str, Dict]:
    """
    Generate a classification report for binary link prediction.
    
    Args:
        predictions: Predicted scores
        targets: Ground truth binary labels
        threshold: Threshold for converting scores to binary predictions
        output_dict: Whether to return a dictionary or string
        
    Returns:
        Classification report as a dictionary or string
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert scores to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Generate classification report
    return sklearn_classification_report(targets, binary_preds, output_dict=output_dict)

def best_threshold(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find the best threshold to maximize a specified metric.
    
    Args:
        predictions: Predicted scores
        targets: Ground truth binary labels
        metric: Metric to optimize ('f1', 'accuracy')
        
    Returns:
        Tuple of (best threshold, best metric value)
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Generate thresholds to try (100 values between min and max of predictions)
    min_pred = predictions.min()
    max_pred = predictions.max()
    thresholds = np.linspace(min_pred, max_pred, 100)
    
    best_score = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        binary_preds = (predictions >= thresh).astype(int)
        
        if metric == 'f1':
            score = sklearn_f1_score(targets, binary_preds)
        elif metric == 'accuracy':
            score = accuracy_score(targets, binary_preds)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score
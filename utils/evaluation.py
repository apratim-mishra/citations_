from typing import Union, Dict, List, Tuple, Optional, TYPE_CHECKING
import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error, max_error
)

# To avoid circular imports, use TYPE_CHECKING for imports used only in type annotations
if TYPE_CHECKING:
    from models.gnn_models import RCRPredictionGNN, MultiTaskGNN

# For runtime isinstance checks, we need to import the actual classes
# Create empty placeholder classes if necessary to avoid ImportError
class RCRPredictionGNN:
    pass

class MultiTaskGNN:
    pass

# Try to import the actual classes, but don't fail if not possible
try:
    from models.gnn_models import RCRPredictionGNN, MultiTaskGNN
except ImportError:
    pass

def compute_metrics(
    model, 
    graph: HeteroData,
    test_data: Tuple[torch.Tensor, torch.Tensor],
    k_values: List[int] = [10],
    device: torch.device = None
) -> Dict[str, float]:
    """
    Compute metrics for link prediction evaluation.
    
    Args:
        model: Link prediction model
        graph: HeteroData graph
        test_data: Test data (edge_label_index, edge_label)
        k_values: List of k values for precision@k
        device: Computation device
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Implementation needed
    return {}

def evaluate_regression(
    model: Union['RCRPredictionGNN', 'MultiTaskGNN'],
    graph: HeteroData,
    node_indices: torch.Tensor,
    true_values: torch.Tensor,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate node regression performance.
    
    Args:
        model: Regression model
        graph: HeteroData graph
        node_indices: Indices of nodes to evaluate
        true_values: True values for the nodes
        device: Computation device
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.eval()
    
    # Move data to device
    graph = graph.to(device)
    node_indices = node_indices.to(device)
    true_values = true_values.to(device)
    
    with torch.no_grad():
        # Get predictions based on model type
        if isinstance(model, RCRPredictionGNN):
            # RCRPredictionGNN predicts for all papers directly
            all_predictions = model(graph.x_dict, graph.edge_index_dict)
            predictions = all_predictions[node_indices]
        elif isinstance(model, MultiTaskGNN):
            # MultiTaskGNN has a predict_rcr method
            all_predictions = model.predict_rcr(graph.x_dict, graph.edge_index_dict)
            predictions = all_predictions[node_indices]
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    
    # Move to CPU for metrics calculation
    predictions = predictions.cpu().numpy()
    true_values = true_values.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # Calculate additional metrics
    rmse = np.sqrt(mse)
    
    # Calculate explained variance
    explained_variance = explained_variance_score(true_values, predictions)
    
    # Calculate median absolute error
    median_ae = median_absolute_error(true_values, predictions)
    
    # Calculate maximum error
    max_error = max_error(true_values, predictions)
    
    # Create metrics dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'median_ae': median_ae,
        'max_error': max_error,
        'r2': r2,
        'explained_variance': explained_variance
    }
    
    return metrics


def evaluate_multitask(
    model: MultiTaskGNN,
    graph: HeteroData,
    link_test_data: Tuple[torch.Tensor, torch.Tensor],
    regression_test_indices: torch.Tensor,
    regression_test_values: torch.Tensor,
    k_values: List[int] = [10],
    device: torch.device = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multi-task model on both link prediction and regression.
    
    Args:
        model: Multi-task model
        graph: HeteroData graph
        link_test_data: Link prediction test data (edge_label_index, edge_label)
        regression_test_indices: Indices of test nodes for regression
        regression_test_values: True values for test nodes
        k_values: List of k values for link prediction metrics
        device: Computation device
        
    Returns:
        Dictionary of link prediction and regression metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate link prediction
    link_metrics = compute_metrics(
        model=model,
        graph=graph,
        test_data=link_test_data,
        k_values=k_values,
        device=device
    )
    
    # Evaluate regression
    regression_metrics = evaluate_regression(
        model=model,
        graph=graph,
        node_indices=regression_test_indices,
        true_values=regression_test_values,
        device=device
    )
    
    # Combine metrics
    metrics = {
        'link_prediction': link_metrics,
        'regression': regression_metrics
    }
    
    return metrics

def validate(
    model,
    graph: HeteroData,
    validation_data,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Validate model performance on validation data.
    
    Args:
        model: Model to validate
        graph: HeteroData graph
        validation_data: Validation data
        device: Computation device
        
    Returns:
        Dictionary of validation metrics
    """
    # Implement validation logic
    return {}

def test(
    model,
    graph: HeteroData,
    test_data,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Test model performance on test data.
    
    Args:
        model: Model to test
        graph: HeteroData graph
        test_data: Test data
        device: Computation device
        
    Returns:
        Dictionary of test metrics
    """
    # Implement test logic
    return {}

def roc_auc_test(
    model,
    graph: HeteroData,
    test_data,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Calculate ROC AUC for model on test data.
    
    Args:
        model: Model to test
        graph: HeteroData graph
        test_data: Test data
        device: Computation device
        
    Returns:
        Dictionary with ROC AUC score
    """
    # Implement ROC AUC calculation
    return {}

def precision_recall_curve(
    model,
    graph: HeteroData,
    test_data,
    device: torch.device = None
) -> Tuple:
    """
    Calculate precision-recall curve for model on test data.
    
    Args:
        model: Model to test
        graph: HeteroData graph
        test_data: Test data
        device: Computation device
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Implement precision-recall curve calculation
    return ([], [], [])
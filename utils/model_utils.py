"""
Utility functions for model handling.
"""
import os
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch_geometric.data import HeteroData

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_size(model: torch.nn.Module) -> float:
    """
    Get the size of a model in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024**2  # MB
    return total_size

def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model checkpoint with optional metadata.
    
    Args:
        model: PyTorch model
        path: Path to save checkpoint
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Optional epoch number
        metrics: Optional dictionary of evaluation metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'class_name': model.__class__.__name__,
            'params': {
                # Add model-specific parameters here if needed
            }
        }
    }
    
    # Add optional metadata
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Save checkpoint
    torch.save(checkpoint, path)

def load_model(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load model onto
        
    Returns:
        Tuple of (loaded model, checkpoint metadata)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state dict if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state dict if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Extract metadata
    metadata = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}
    
    return model, metadata

def freeze_parameters(
    model: torch.nn.Module,
    module_names: Optional[List[str]] = None
) -> None:
    """
    Freeze model parameters to prevent updates during training.
    
    Args:
        model: PyTorch model
        module_names: Optional list of module names to freeze (freezes all if None)
    """
    if module_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specified modules
        for name, module in model.named_children():
            if name in module_names:
                for param in module.parameters():
                    param.requires_grad = False

def unfreeze_parameters(
    model: torch.nn.Module,
    module_names: Optional[List[str]] = None
) -> None:
    """
    Unfreeze model parameters to allow updates during training.
    
    Args:
        model: PyTorch model
        module_names: Optional list of module names to unfreeze (unfreezes all if None)
    """
    if module_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze specified modules
        for name, module in model.named_children():
            if name in module_names:
                for param in module.parameters():
                    param.requires_grad = True

def gradual_unfreezing(
    model: torch.nn.Module,
    epoch: int,
    unfreeze_schedule: Dict[int, List[str]]
) -> None:
    """
    Gradually unfreeze model parameters based on epoch.
    
    Args:
        model: PyTorch model
        epoch: Current epoch
        unfreeze_schedule: Dictionary mapping epochs to module names to unfreeze
    """
    # Check if any modules should be unfrozen at the current epoch
    if epoch in unfreeze_schedule:
        modules_to_unfreeze = unfreeze_schedule[epoch]
        unfreeze_parameters(model, modules_to_unfreeze)
        print(f"Epoch {epoch}: Unfroze modules {modules_to_unfreeze}")

def export_onnx(
    model: torch.nn.Module,
    path: str,
    input_sample: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        path: Path to save ONNX model
        input_sample: Sample input(s) for tracing
        input_names: Names of input tensors
        output_names: Names of output tensors
        dynamic_axes: Optional dict specifying dynamic axes
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Export model
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_sample,
            path,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=12
        )
    
    print(f"Model exported to {path}")
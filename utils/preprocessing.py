import torch
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from torch_geometric.data import HeteroData

def split_node_data(
    graph: HeteroData, 
    node_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split nodes into train, validation, and test sets.
    
    Args:
        graph: HeteroData graph
        node_type: Type of nodes to split
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        test_ratio: Fraction of nodes for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of boolean masks (train_mask, val_mask, test_mask)
    """
    if not node_type in graph.node_types:
        raise ValueError(f"Node type '{node_type}' not found in graph")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get number of nodes
    num_nodes = graph[node_type].num_nodes
    
    # Create random permutation of node indices
    perm = torch.randperm(num_nodes)
    
    # Calculate split sizes
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Assign nodes to splits
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask

def split_edge_data(
    graph: HeteroData,
    edge_type: Tuple[str, str, str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edges into train, validation, and test sets.
    
    Args:
        graph: HeteroData graph
        edge_type: Edge type as tuple (src_node_type, edge_type, dst_node_type)
        train_ratio: Fraction of edges for training
        val_ratio: Fraction of edges for validation
        test_ratio: Fraction of edges for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of boolean masks (train_mask, val_mask, test_mask)
    """
    if not edge_type in graph.edge_types:
        raise ValueError(f"Edge type '{edge_type}' not found in graph")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get number of edges
    num_edges = graph[edge_type].edge_index.size(1)
    
    # Create random permutation of edge indices
    perm = torch.randperm(num_edges)
    
    # Calculate split sizes
    train_size = int(train_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    # Create masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    # Assign edges to splits
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask

def save_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor to disk."""
    torch.save(tensor, path)

def load_tensor(path: str) -> torch.Tensor:
    """Load a tensor from disk."""
    return torch.load(path)
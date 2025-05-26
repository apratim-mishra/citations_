"""
Distance metrics for evaluating node embeddings.
"""
import torch
import numpy as np
from typing import Union, Optional, Tuple

def cosine_similarity(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate cosine similarity between vectors.
    
    Args:
        x: First vector or batch of vectors
        y: Second vector or batch of vectors
        
    Returns:
        Cosine similarity (higher values indicate more similar vectors)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # PyTorch implementation
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
        return torch.sum(x_norm * y_norm, dim=-1)
    else:
        # NumPy implementation
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True)
        return np.sum(x_norm * y_norm, axis=-1)

def euclidean_distance(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate Euclidean distance between vectors.
    
    Args:
        x: First vector or batch of vectors
        y: Second vector or batch of vectors
        
    Returns:
        Euclidean distance (lower values indicate more similar vectors)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # PyTorch implementation
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))
    else:
        # NumPy implementation
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

def manhattan_distance(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate Manhattan (L1) distance between vectors.
    
    Args:
        x: First vector or batch of vectors
        y: Second vector or batch of vectors
        
    Returns:
        Manhattan distance (lower values indicate more similar vectors)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # PyTorch implementation
        return torch.sum(torch.abs(x - y), dim=-1)
    else:
        # NumPy implementation
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        return np.sum(np.abs(x - y), axis=-1)

def pairwise_distances(
    embeddings: Union[torch.Tensor, np.ndarray],
    metric: str = 'cosine'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate pairwise distances/similarities between all embeddings.
    
    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        metric: Distance metric to use ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        Pairwise distance/similarity matrix [num_nodes, num_nodes]
    """
    if isinstance(embeddings, torch.Tensor):
        # PyTorch implementation
        if metric == 'cosine':
            # Normalize embeddings
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # Calculate pairwise cosine similarities
            return torch.mm(embeddings_norm, embeddings_norm.t())
        elif metric == 'euclidean':
            # Calculate pairwise Euclidean distances
            return torch.cdist(embeddings, embeddings, p=2)
        elif metric == 'manhattan':
            # Calculate pairwise Manhattan distances
            return torch.cdist(embeddings, embeddings, p=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    else:
        # NumPy implementation
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        if metric == 'cosine':
            # Normalize embeddings
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Calculate pairwise cosine similarities
            return np.dot(embeddings_norm, embeddings_norm.T)
        elif metric == 'euclidean':
            # Calculate pairwise Euclidean distances
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(embeddings)
        elif metric == 'manhattan':
            # Calculate pairwise Manhattan distances
            from sklearn.metrics.pairwise import manhattan_distances
            return manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
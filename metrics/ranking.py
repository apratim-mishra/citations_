"""
Ranking metrics for recommendation system evaluation.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

def precision_at_k(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k: int
) -> float:
    """
    Calculate Precision@k for recommendation.
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k: Number of recommendations to consider
        
    Returns:
        Precision@k score
    """
    # Ensure predictions only include up to k items
    predictions = predictions[:, :k]
    
    # Convert to sets for intersection calculation
    prediction_sets = [set(pred.cpu().numpy()) for pred in predictions]
    ground_truth_sets = [set(gt.cpu().numpy()) for gt in ground_truth]
    
    # Calculate precision for each user
    precisions = []
    for pred_set, gt_set in zip(prediction_sets, ground_truth_sets):
        if not gt_set:  # Skip users with no ground truth
            continue
        
        # Calculate precision: |relevant ∩ retrieved| / |retrieved|
        precisions.append(len(pred_set.intersection(gt_set)) / len(pred_set))
    
    # Average precision across users
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k: int
) -> float:
    """
    Calculate Recall@k for recommendation.
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k: Number of recommendations to consider
        
    Returns:
        Recall@k score
    """
    # Ensure predictions only include up to k items
    predictions = predictions[:, :k]
    
    # Convert to sets for intersection calculation
    prediction_sets = [set(pred.cpu().numpy()) for pred in predictions]
    ground_truth_sets = [set(gt.cpu().numpy()) for gt in ground_truth]
    
    # Calculate recall for each user
    recalls = []
    for pred_set, gt_set in zip(prediction_sets, ground_truth_sets):
        if not gt_set:  # Skip users with no ground truth
            continue
        
        # Calculate recall: |relevant ∩ retrieved| / |relevant|
        recalls.append(len(pred_set.intersection(gt_set)) / len(gt_set))
    
    # Average recall across users
    return np.mean(recalls) if recalls else 0.0

def mean_average_precision(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Average Precision (MAP) for recommendation.
    
    Args:
        predictions: Predicted item indices (batch_size, pred_length)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k: Optional maximum number of predictions to consider
        
    Returns:
        MAP score
    """
    if k is not None:
        predictions = predictions[:, :k]
    
    # Calculate average precision for each user
    average_precisions = []
    for i in range(len(predictions)):
        # Get current user's predictions and ground truth
        user_preds = predictions[i]
        user_gt = set(ground_truth[i].cpu().numpy())
        
        if not user_gt:  # Skip users with no ground truth
            continue
        
        # Compute relevance for each position (1 if item is relevant, 0 otherwise)
        relevance = torch.tensor([1 if item.item() in user_gt else 0 
                                 for item in user_preds])
        
        if relevance.sum() == 0:  # No relevant items retrieved
            average_precisions.append(0.0)
            continue
        
        # Calculate cumulative sum of relevant items
        cumulative_relevance = torch.cumsum(relevance, dim=0)
        
        # Calculate precision at each relevant position
        precision_at_relevant = cumulative_relevance / torch.arange(1, len(user_preds) + 1)
        
        # Only consider precision at relevant positions
        precision_at_relevant = precision_at_relevant * relevance
        
        # Calculate average precision
        ap = precision_at_relevant.sum() / min(len(user_gt), len(user_preds))
        average_precisions.append(ap.item())
    
    # Return mean of average precisions
    return np.mean(average_precisions) if average_precisions else 0.0

def ndcg_at_k(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k: int,
    use_graded_relevance: bool = False,
    relevance_scores: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        predictions: Predicted item indices (batch_size, pred_length)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k: Number of recommendations to consider
        use_graded_relevance: Whether to use graded relevance scores
        relevance_scores: Optional tensor of relevance scores for ground truth items
        
    Returns:
        NDCG@k score
    """
    # Ensure predictions only include up to k items
    predictions = predictions[:, :k]
    
    ndcg_scores = []
    for i in range(len(predictions)):
        # Get current user's predictions and ground truth
        user_preds = predictions[i].cpu().numpy()
        user_gt = ground_truth[i].cpu().numpy()
        
        if len(user_gt) == 0:  # Skip users with no ground truth
            continue
        
        # Create a relevance array (default: binary relevance)
        if use_graded_relevance and relevance_scores is not None:
            # Map predictions to their relevance scores
            user_rel_scores = relevance_scores[i].cpu().numpy()
            # Create a mapping from item ID to relevance score
            item_to_relevance = {item: score for item, score in zip(user_gt, user_rel_scores)}
            # Get relevance for each predicted item (0 if not in ground truth)
            relevance = np.array([item_to_relevance.get(item, 0.0) for item in user_preds])
        else:
            # Binary relevance: 1 if item is in ground truth, 0 otherwise
            relevance = np.array([1.0 if item in user_gt else 0.0 for item in user_preds])
        
        # If no relevant items were recommended
        if np.sum(relevance) == 0:
            ndcg_scores.append(0.0)
            continue
        
        # Calculate DCG: sum of (rel_i / log2(i+1))
        discounts = np.log2(np.arange(2, len(relevance) + 2))  # [log2(2), log2(3), ...]
        dcg = np.sum(relevance / discounts)
        
        # Calculate ideal DCG (IDCG)
        # Sort relevance in descending order for the ideal ranking
        if use_graded_relevance and relevance_scores is not None:
            # Get all relevance scores and sort
            ideal_relevance = np.sort(list(item_to_relevance.values()))[::-1]
            # Truncate to k items
            ideal_relevance = ideal_relevance[:k]
        else:
            # For binary relevance, IDCG is just sum of 1/log2(i+1) for min(k, num_relevant) positions
            ideal_relevance = np.ones(min(k, len(user_gt)))
        
        ideal_discounts = np.log2(np.arange(2, len(ideal_relevance) + 2))
        idcg = np.sum(ideal_relevance / ideal_discounts)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    # Return mean NDCG
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def mean_reciprocal_rank(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for recommendation.
    
    Args:
        predictions: Predicted item indices (batch_size, pred_length)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k: Optional maximum number of predictions to consider
        
    Returns:
        MRR score
    """
    if k is not None:
        predictions = predictions[:, :k]
    
    reciprocal_ranks = []
    for i in range(len(predictions)):
        # Get current user's predictions and ground truth
        user_preds = predictions[i].cpu().numpy()
        user_gt = set(ground_truth[i].cpu().numpy())
        
        if not user_gt:  # Skip users with no ground truth
            continue
        
        # Find the rank of the first relevant item
        for rank, item in enumerate(user_preds):
            if item in user_gt:
                # Reciprocal rank is 1/(rank+1) because ranks are 0-indexed
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            # No relevant items found in the predictions
            reciprocal_ranks.append(0.0)
    
    # Return mean of reciprocal ranks
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_all_ranking_metrics(
    predictions: torch.Tensor, 
    ground_truth: torch.Tensor, 
    k_values: List[int] = [5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Compute multiple ranking metrics at different k values.
    
    Args:
        predictions: Predicted item indices (batch_size, pred_length)
        ground_truth: Ground truth item indices (batch_size, variable length)
        k_values: List of k values for which to compute metrics
        
    Returns:
        Dictionary of metric names to values
    """
    results = {}
    
    # Ensure we have enough predictions for the largest k
    max_k = max(k_values)
    if predictions.shape[1] < max_k:
        print(f"Warning: predictions only contains {predictions.shape[1]} items, but max k is {max_k}")
    
    # Compute metrics at each k value
    for k in k_values:
        if predictions.shape[1] >= k:
            results[f'precision@{k}'] = precision_at_k(predictions, ground_truth, k)
            results[f'recall@{k}'] = recall_at_k(predictions, ground_truth, k)
            results[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth, k)
    
    # Compute k-independent metrics
    results['map'] = mean_average_precision(predictions, ground_truth)
    results['mrr'] = mean_reciprocal_rank(predictions, ground_truth)
    
    return results
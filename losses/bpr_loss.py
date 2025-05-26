"""
Bayesian Personalized Ranking (BPR) loss implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss for implicit feedback recommendation.
    
    BPR optimizes the ranking of items by maximizing the difference between 
    positive and negative item scores, encouraging the model to rank positive 
    items higher than negative ones.
    """
    
    def __init__(self, reduction: str = 'mean', weight: Optional[float] = None):
        """
        Initialize BPR loss.
        
        Args:
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
            weight: Optional weight to apply to the loss
        """
        super(BPRLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
    
    def forward(
        self, 
        pos_scores: torch.Tensor, 
        neg_scores: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate BPR loss.
        
        Args:
            pos_scores: Scores for positive samples (batch_size,)
            neg_scores: Scores for negative samples (batch_size, n_neg) or (batch_size,)
            weights: Optional sample weights (batch_size,)
            
        Returns:
            BPR loss
        """
        # Check if neg_scores is a batch of multiple negatives or just a single negative per positive
        if neg_scores.dim() > pos_scores.dim():
            # Multiple negatives per positive (batch_size, n_neg)
            pos_scores = pos_scores.unsqueeze(1)  # (batch_size, 1)
            # Compute difference for each positive-negative pair
            diff = pos_scores - neg_scores  # (batch_size, n_neg)
        else:
            # Single negative per positive
            diff = pos_scores - neg_scores  # (batch_size,)
        
        # Apply negative log sigmoid to the differences
        loss = -F.logsigmoid(diff)
        
        # Apply sample weights if provided
        if weights is not None:
            if weights.dim() < loss.dim():
                weights = weights.unsqueeze(1)  # (batch_size, 1)
            loss = loss * weights
        
        # Apply user-provided weight multiplier
        if self.weight is not None:
            loss = loss * self.weight
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

# More sophisticated version with regularization and margin
class EnhancedBPRLoss(nn.Module):
    """
    Enhanced Bayesian Personalized Ranking loss with regularization and margin.
    """
    
    def __init__(
        self, 
        margin: float = 1.0,
        user_reg: float = 0.0,
        item_reg: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Enhanced BPR loss.
        
        Args:
            margin: Minimum desired difference between positive and negative scores
            user_reg: L2 regularization coefficient for user embeddings
            item_reg: L2 regularization coefficient for item embeddings
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
        """
        super(EnhancedBPRLoss, self).__init__()
        self.margin = margin
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.reduction = reduction
    
    def forward(
        self, 
        pos_scores: torch.Tensor, 
        neg_scores: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        item_pos_embeddings: Optional[torch.Tensor] = None,
        item_neg_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate enhanced BPR loss with regularization.
        
        Args:
            pos_scores: Scores for positive samples (batch_size,)
            neg_scores: Scores for negative samples (batch_size, n_neg) or (batch_size,)
            user_embeddings: User embeddings for regularization
            item_pos_embeddings: Positive item embeddings for regularization
            item_neg_embeddings: Negative item embeddings for regularization
            
        Returns:
            Dictionary containing total loss and component losses
        """
        # Check if neg_scores is a batch of multiple negatives or just a single negative per positive
        if neg_scores.dim() > pos_scores.dim():
            # Multiple negatives per positive (batch_size, n_neg)
            pos_scores = pos_scores.unsqueeze(1)  # (batch_size, 1)
        
        # Compute difference with margin
        diff = pos_scores - neg_scores - self.margin
        
        # Apply negative log sigmoid to the differences with softplus for numerical stability
        # softplus(-x) is a numerically stable version of -log(sigmoid(x))
        ranking_loss = F.softplus(-diff)
        
        # Compute regularization loss if embeddings are provided
        reg_loss = 0.0
        if self.user_reg > 0 and user_embeddings is not None:
            reg_loss += self.user_reg * user_embeddings.norm(p=2, dim=1).pow(2).mean()
        
        if self.item_reg > 0:
            if item_pos_embeddings is not None:
                reg_loss += self.item_reg * item_pos_embeddings.norm(p=2, dim=1).pow(2).mean()
            
            if item_neg_embeddings is not None:
                if item_neg_embeddings.dim() > 2:  # For multiple negatives
                    reg_loss += self.item_reg * item_neg_embeddings.norm(p=2, dim=2).pow(2).mean()
                else:
                    reg_loss += self.item_reg * item_neg_embeddings.norm(p=2, dim=1).pow(2).mean()
        
        # Apply reduction to ranking loss
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            ranking_loss = ranking_loss.mean()
        elif self.reduction == 'sum':
            ranking_loss = ranking_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        
        # Total loss
        total_loss = ranking_loss + reg_loss
        
        return {
            'loss': total_loss,
            'ranking_loss': ranking_loss,
            'reg_loss': reg_loss
        }
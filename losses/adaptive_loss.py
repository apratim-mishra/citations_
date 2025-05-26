"""
Adaptive loss functions for recommendation systems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple

class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance in recommendation systems.
    
    Focal loss applies a modulating factor to cross-entropy loss to focus more
    on hard examples while down-weighting easy ones.
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for the rare class (positive samples)
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        pred_scores: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            pred_scores: Predicted scores (batch_size,)
            targets: Target binary labels (batch_size,)
            
        Returns:
            Focal loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(pred_scores)
        
        # Calculate binary cross entropy term
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores, targets, reduction='none'
        )
        
        # Calculate the focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Apply focal weight to BCE loss
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class AdaptiveHybridLoss(nn.Module):
    """
    Hybrid loss function that combines BPR and BCE elements with adaptive weighting.
    """
    
    def __init__(
        self, 
        alpha: float = 0.5, 
        gamma: float = 2.0,
        beta: float = 0.5,  # Balance between BPR and BCE
        margin: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Adaptive Hybrid Loss.
        
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            beta: Balance weight between BPR (beta) and BCE (1-beta)
            margin: Margin for BPR component
            reduction: Specifies the reduction to apply to the output
        """
        super(AdaptiveHybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.margin = margin
        self.reduction = reduction
        
        # Initialize individual loss components
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
    
    def forward(
        self, 
        pos_scores: torch.Tensor, 
        neg_scores: torch.Tensor,
        pos_targets: Optional[torch.Tensor] = None, 
        neg_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate adaptive hybrid loss.
        
        Args:
            pos_scores: Scores for positive samples (batch_size,)
            neg_scores: Scores for negative samples (batch_size, n_neg) or (batch_size,)
            pos_targets: Target values for positive samples (defaults to ones)
            neg_targets: Target values for negative samples (defaults to zeros)
            
        Returns:
            Dictionary containing total loss and component losses
        """
        # Default targets if not provided
        if pos_targets is None:
            pos_targets = torch.ones_like(pos_scores)
        if neg_targets is None:
            if neg_scores.dim() > 1:
                neg_targets = torch.zeros_like(neg_scores)
            else:
                neg_targets = torch.zeros_like(neg_scores)
                
        # Determine if using multiple negatives
        multi_neg = neg_scores.dim() > pos_scores.dim()
        
        # --- BPR Component ---
        if multi_neg:
            # Handle multiple negatives per positive
            pos_expanded = pos_scores.unsqueeze(1)  # (batch_size, 1)
            # Compute difference for each positive-negative pair
            diff = pos_expanded - neg_scores - self.margin  # (batch_size, n_neg)
            bpr_loss = F.softplus(-diff)  # Numerically stable version of -log(sigmoid(diff))
            
            if self.reduction != 'none':
                bpr_loss = bpr_loss.mean(dim=1)  # Average over negatives
        else:
            # Single negative per positive
            diff = pos_scores - neg_scores - self.margin
            bpr_loss = F.softplus(-diff)
            
        # --- BCE Component with Focal Weighting ---
        # For positive samples
        bce_pos_loss = self.focal_loss(pos_scores, pos_targets)
        
        # For negative samples
        if multi_neg:
            bce_neg_loss = self.focal_loss(neg_scores.view(-1), neg_targets.view(-1))
            if self.reduction != 'none':
                bce_neg_loss = bce_neg_loss.view(neg_scores.shape).mean(dim=1)
        else:
            bce_neg_loss = self.focal_loss(neg_scores, neg_targets)
        
        # Combine BCE losses
        bce_loss = bce_pos_loss + bce_neg_loss
            
        # Combine BPR and BCE with adaptive weighting
        combined_loss = self.beta * bpr_loss + (1 - self.beta) * bce_loss
        
        # Apply reduction
        if self.reduction == 'none':
            total_loss = combined_loss
        elif self.reduction == 'mean':
            total_loss = combined_loss.mean()
        elif self.reduction == 'sum':
            total_loss = combined_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
            
        # Return loss components for logging
        return {
            'loss': total_loss,
            'bpr_loss': bpr_loss.mean() if self.reduction != 'none' else bpr_loss,
            'bce_loss': bce_loss.mean() if self.reduction != 'none' else bce_loss,
        }

class AdaptiveMarginLoss(nn.Module):
    """
    Adaptive margin loss that adjusts the margin based on item popularity.
    """
    def __init__(
        self,
        base_margin: float = 0.5,
        popularity_factor: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize Adaptive Margin Loss.
        
        Args:
            base_margin: Base margin value
            popularity_factor: Factor to adjust margin based on popularity
            reduction: Specifies the reduction to apply to the output
        """
        super(AdaptiveMarginLoss, self).__init__()
        self.base_margin = base_margin
        self.popularity_factor = popularity_factor
        self.reduction = reduction
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        pos_item_popularity: torch.Tensor,
        neg_item_popularity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate adaptive margin loss.
        
        Args:
            pos_scores: Scores for positive samples (batch_size,)
            neg_scores: Scores for negative samples (batch_size, n_neg) or (batch_size,)
            pos_item_popularity: Popularity score for positive items (batch_size,)
            neg_item_popularity: Popularity score for negative items
            
        Returns:
            Loss value
        """
        # Adjust margin based on popularity
        # More popular items have higher popularity scores (e.g., normalized counts)
        # We want to enforce a larger margin for less popular items (harder positives)
        adjusted_margin = self.base_margin + self.popularity_factor * (1 - pos_item_popularity)
        
        # Expand dimensions if needed
        if neg_scores.dim() > pos_scores.dim():
            pos_scores = pos_scores.unsqueeze(1)
            adjusted_margin = adjusted_margin.unsqueeze(1)
        
        # Compute margin-based ranking loss
        diff = pos_scores - neg_scores - adjusted_margin
        loss = F.softplus(-diff)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
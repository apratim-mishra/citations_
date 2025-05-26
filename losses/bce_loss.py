"""
Binary Cross Entropy loss implementations for link prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class BCELoss(nn.Module):
    """
    Standard Binary Cross Entropy Loss with options for link prediction.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize BCE loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            pos_weight: Weight for positive examples (useful for imbalanced data)
        """
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate BCE loss.
        
        Args:
            pred: Predicted scores [batch_size]
            target: Target labels [batch_size]
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE loss that can apply different weights to positive and negative examples.
    """
    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize weighted BCE loss.
        
        Args:
            pos_weight: Weight for positive examples
            neg_weight: Weight for negative examples
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weighted BCE loss.
        
        Args:
            pred: Predicted scores [batch_size]
            target: Target labels [batch_size]
            
        Returns:
            Loss value
        """
        # Apply sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Binary cross entropy (manual calculation)
        bce = -target * torch.log(pred_probs + 1e-7) - (1 - target) * torch.log(1 - pred_probs + 1e-7)
        
        # Apply weights
        weights = target * self.pos_weight + (1 - target) * self.neg_weight
        weighted_bce = weights * bce
        
        # Apply reduction
        if self.reduction == 'none':
            return weighted_bce
        elif self.reduction == 'mean':
            return weighted_bce.mean()
        elif self.reduction == 'sum':
            return weighted_bce.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
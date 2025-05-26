"""
Loss functions for link prediction and regression tasks.
"""
from .bce_loss import BCELoss, WeightedBCELoss
from .bpr_loss import BPRLoss, EnhancedBPRLoss
from .contrastive_loss import ContrastiveLoss, TripletLoss
from .adaptive_loss import FocalLoss, AdaptiveHybridLoss, AdaptiveMarginLoss

__all__ = [
    'BCELoss',
    'WeightedBCELoss',
    'BPRLoss',
    'EnhancedBPRLoss',
    'ContrastiveLoss',
    'TripletLoss',
    'FocalLoss',
    'AdaptiveHybridLoss',
    'AdaptiveMarginLoss'
]
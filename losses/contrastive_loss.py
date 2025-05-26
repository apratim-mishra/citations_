"""
Contrastive loss implementations for link prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for link prediction.
    
    Creates a loss that encourages the embedding vectors of positive pairs to be close
    and embedding vectors of negative pairs to be far apart.
    """
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs (larger values enforce stronger separation)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss.
        
        Args:
            src_embeddings: Source node embeddings [batch_size, embedding_dim]
            dst_embeddings: Destination node embeddings [batch_size, embedding_dim]
            target: Binary labels (1 for positive pairs, 0 for negative pairs) [batch_size]
            
        Returns:
            Contrastive loss
        """
        # Calculate Euclidean distance
        distance = F.pairwise_distance(src_embeddings, dst_embeddings, p=2)
        
        # Contrastive loss components
        # For positive pairs: encourage embeddings to be close (smaller distance)
        positive_loss = target * distance.pow(2)
        
        # For negative pairs: encourage embeddings to be far apart (distance > margin)
        # If distance > margin, loss is 0
        negative_loss = (1 - target) * F.relu(self.margin - distance).pow(2)
        
        # Combine losses
        loss = positive_loss + negative_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class TripletLoss(nn.Module):
    """
    Triplet loss for link prediction.
    
    Creates a loss using triplets of (anchor, positive, negative) examples.
    The loss encourages the anchor to be closer to the positive than to the negative.
    """
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
        distance: str = 'euclidean'
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin to enforce between positive and negative pairs
            reduction: Reduction method ('none', 'mean', 'sum')
            distance: Distance metric ('euclidean', 'cosine')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = distance
        
        if distance not in ['euclidean', 'cosine']:
            raise ValueError(f"Unsupported distance metric: {distance}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss
        """
        if self.distance == 'euclidean':
            # Calculate Euclidean distances
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            
            # Triplet loss: encourage pos_dist to be smaller than neg_dist
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        elif self.distance == 'cosine':
            # Calculate cosine similarity (higher means more similar)
            pos_sim = F.cosine_similarity(anchor, positive)
            neg_sim = F.cosine_similarity(anchor, negative)
            
            # Triplet loss: encourage pos_sim to be higher than neg_sim
            # Higher similarity should result in lower loss, so we negate
            loss = F.relu(neg_sim - pos_sim + self.margin)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
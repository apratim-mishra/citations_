"""
Sampling utilities for link prediction.
"""
import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from torch_geometric.data import HeteroData

class HardNegativeSampler:
    """
    Hard negative sampler for link prediction.
    Uses a reference model to identify challenging negative examples.
    """
    def __init__(
        self,
        graph: HeteroData,
        ref_model: torch.nn.Module,
        neg_ratio: int = 5,
        device: Optional[torch.device] = None,
        edge_type: Tuple[str, str, str] = ('paper', 'cites', 'paper')
    ):
        """
        Initialize hard negative sampler.
        
        Args:
            graph: HeteroData graph
            ref_model: Reference model for scoring potential negative edges
            neg_ratio: Number of negative samples per positive sample
            device: Computation device
            edge_type: Type of edges to generate negative samples for
        """
        self.graph = graph
        self.ref_model = ref_model
        self.neg_ratio = neg_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_type = edge_type
        
        # Extract node types from edge type
        self.src_node_type, _, self.dst_node_type = edge_type
        
        # Set model to evaluation mode
        self.ref_model.eval()
        
        # Cache of existing edges for faster lookup
        self._build_edge_cache()
    
    def _build_edge_cache(self):
        """Build a cache of existing edges for faster negative sampling."""
        edge_index = self.graph[self.edge_type].edge_index
        self.edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    
    def sample_negatives(
        self,
        batch: HeteroData,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        neg_ratio: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hard negative edges using the reference model.
        
        Args:
            batch: Mini-batch of heterogeneous graph
            edge_label_index: Edge indices for which to sample negatives
            edge_label: Edge labels (0 or 1)
            neg_ratio: Number of negative samples per positive (overrides default)
            
        Returns:
            Tuple of (augmented_edge_label_index, augmented_edge_label)
        """
        if neg_ratio is None:
            neg_ratio = self.neg_ratio
        
        # Find positive edges
        pos_mask = edge_label == 1
        pos_edge_index = edge_label_index[:, pos_mask]
        
        # Number of positive edges
        num_pos = pos_edge_index.size(1)
        
        # Number of negative samples to generate
        num_neg = num_pos * neg_ratio
        
        # Get total number of nodes for source and destination types
        num_src_nodes = batch[self.src_node_type].num_nodes
        num_dst_nodes = batch[self.dst_node_type].num_nodes
        
        # Generate candidate negative edges
        num_candidates = num_neg * 5  # Generate more candidates than needed
        
        # Sample source and destination nodes
        src_nodes = torch.randint(0, num_src_nodes, (num_candidates,), device=self.device)
        dst_nodes = torch.randint(0, num_dst_nodes, (num_candidates,), device=self.device)
        
        # Create candidate edge index
        candidate_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        
        # Filter out existing edges
        valid_candidates = []
        for i in range(candidate_edge_index.size(1)):
            src, dst = candidate_edge_index[0, i].item(), candidate_edge_index[1, i].item()
            if (src, dst) not in self.edge_set:
                valid_candidates.append(i)
        
        # If not enough valid candidates, generate more
        while len(valid_candidates) < num_neg:
            # Generate more candidates
            src_nodes = torch.randint(0, num_src_nodes, (num_candidates,), device=self.device)
            dst_nodes = torch.randint(0, num_dst_nodes, (num_candidates,), device=self.device)
            candidate_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            
            for i in range(candidate_edge_index.size(1)):
                src, dst = candidate_edge_index[0, i].item(), candidate_edge_index[1, i].item()
                if (src, dst) not in self.edge_set:
                    valid_candidates.append(i)
                    if len(valid_candidates) >= num_neg:
                        break
        
        # Use only needed candidates
        valid_candidates = valid_candidates[:num_neg]
        candidate_edge_index = candidate_edge_index[:, valid_candidates]
        
        # Score candidates using reference model
        with torch.no_grad():
            # Make batch predictions to score candidate edges
            scores = self.ref_model(
                batch.x_dict, 
                batch.edge_index_dict, 
                candidate_edge_index
            )
        
        # Select hard negatives (highest scores among candidates)
        _, hard_neg_indices = torch.topk(scores, num_neg)
        hard_neg_edge_index = candidate_edge_index[:, hard_neg_indices]
        
        # Create augmented edge index and labels
        aug_edge_index = torch.cat([pos_edge_index, hard_neg_edge_index], dim=1)
        aug_edge_label = torch.cat([
            torch.ones(num_pos, device=self.device),
            torch.zeros(num_neg, device=self.device)
        ])
        
        return aug_edge_index, aug_edge_label

class RandomNegativeSampler:
    """
    Random negative sampler for link prediction.
    """
    def __init__(
        self,
        graph: HeteroData,
        neg_ratio: int = 5,
        device: Optional[torch.device] = None,
        edge_type: Tuple[str, str, str] = ('paper', 'cites', 'paper')
    ):
        """
        Initialize random negative sampler.
        
        Args:
            graph: HeteroData graph
            neg_ratio: Number of negative samples per positive sample
            device: Computation device
            edge_type: Type of edges to generate negative samples for
        """
        self.graph = graph
        self.neg_ratio = neg_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_type = edge_type
        
        # Extract node types from edge type
        self.src_node_type, _, self.dst_node_type = edge_type
        
        # Cache of existing edges for faster lookup
        self._build_edge_cache()
    
    def _build_edge_cache(self):
        """Build a cache of existing edges for faster negative sampling."""
        edge_index = self.graph[self.edge_type].edge_index
        self.edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    
    def sample_negatives(
        self,
        batch: HeteroData,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        neg_ratio: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random negative edges.
        
        Args:
            batch: Mini-batch of heterogeneous graph
            edge_label_index: Edge indices for which to sample negatives
            edge_label: Edge labels (0 or 1)
            neg_ratio: Number of negative samples per positive (overrides default)
            
        Returns:
            Tuple of (augmented_edge_label_index, augmented_edge_label)
        """
        if neg_ratio is None:
            neg_ratio = self.neg_ratio
        
        # Find positive edges
        pos_mask = edge_label == 1
        pos_edge_index = edge_label_index[:, pos_mask]
        
        # Number of positive edges
        num_pos = pos_edge_index.size(1)
        
        # Number of negative samples to generate
        num_neg = num_pos * neg_ratio
        
        # Get total number of nodes for source and destination types
        num_src_nodes = batch[self.src_node_type].num_nodes
        num_dst_nodes = batch[self.dst_node_type].num_nodes
        
        # Generate random negative edges
        neg_edge_index = self._sample_random_negative_edges(
            num_src_nodes, num_dst_nodes, num_neg
        )
        
        # Create augmented edge index and labels
        aug_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        aug_edge_label = torch.cat([
            torch.ones(num_pos, device=self.device),
            torch.zeros(num_neg, device=self.device)
        ])
        
        return aug_edge_index, aug_edge_label
    
    def _sample_random_negative_edges(
        self,
        num_src_nodes: int,
        num_dst_nodes: int,
        num_samples: int
    ) -> torch.Tensor:
        """
        Sample random negative edges.
        
        Args:
            num_src_nodes: Number of source nodes
            num_dst_nodes: Number of destination nodes
            num_samples: Number of negative samples to generate
            
        Returns:
            Tensor of negative edge indices
        """
        # Generate candidate negative edges
        num_candidates = num_samples * 2  # Generate more candidates than needed
        
        neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        remaining_samples = num_samples
        
        while remaining_samples > 0:
            # Sample source and destination nodes
            src_nodes = torch.randint(0, num_src_nodes, (num_candidates,), device=self.device)
            dst_nodes = torch.randint(0, num_dst_nodes, (num_candidates,), device=self.device)
            
            # Create candidate edge index
            candidate_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            
            # Filter out existing edges
            valid_candidates = []
            for i in range(candidate_edge_index.size(1)):
                src, dst = candidate_edge_index[0, i].item(), candidate_edge_index[1, i].item()
                if (src, dst) not in self.edge_set:
                    valid_candidates.append(i)
                    if len(valid_candidates) >= remaining_samples:
                        break
            
            # Take needed candidates
            if valid_candidates:
                valid_indices = torch.tensor(valid_candidates, device=self.device)
                valid_edge_index = candidate_edge_index[:, valid_indices]
                
                # Add to negative edge index
                neg_edge_index = torch.cat([neg_edge_index, valid_edge_index], dim=1)
                
                # Update remaining samples
                remaining_samples = num_samples - neg_edge_index.size(1)
        
        return neg_edge_index[:, :num_samples]
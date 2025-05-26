"""
Sentence transformer models for text-based recommendation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict, Sequential, Dropout, LayerNorm, ReLU
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. SentenceTransformerEncoder will not work.")
    SentenceTransformer = None


class TextEmbeddingEncoder(nn.Module):
    """
    Text embedding encoder using a pre-trained sentence transformer.
    """
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        output_dim: Optional[int] = None,
        pooling_mode: str = 'mean',
        max_seq_length: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the text embedding encoder.
        
        Args:
            model_name: Name of the pre-trained sentence transformer model
            output_dim: Output embedding dimension (if None, uses model's default)
            pooling_mode: Pooling strategy ('mean', 'max', 'cls')
            max_seq_length: Maximum sequence length for the model
            device: Computation device
        """
        super(TextEmbeddingEncoder, self).__init__()
        
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers package is required for TextEmbeddingEncoder")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.pooling_mode = pooling_mode
        self.max_seq_length = max_seq_length
        
        # Load the pre-trained model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.max_seq_length = max_seq_length
        
        # Get the embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Optional output projection
        self.output_dim = output_dim
        if output_dim is not None and output_dim != self.embedding_dim:
            self.projection = nn.Linear(self.embedding_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of embeddings (batch_size, embedding_dim)
        """
        # Generate embeddings using the sentence transformer
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Apply optional projection
        embeddings = self.projection(embeddings)
        
        return embeddings


class DualEncoderModel(nn.Module):
    """
    Dual encoder model for text-based recommendation.
    """
    def __init__(
        self,
        user_text_encoder: TextEmbeddingEncoder,
        item_text_encoder: Optional[TextEmbeddingEncoder] = None,
        projection_dim: Optional[int] = None,
        user_metadata_dim: Optional[int] = None,
        item_metadata_dim: Optional[int] = None,
        temperature: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the dual encoder model.
        
        Args:
            user_text_encoder: Text encoder for user text
            item_text_encoder: Text encoder for item text (if None, shares with user)
            projection_dim: Dimension for projection layer (if None, no projection)
            user_metadata_dim: Dimension of user metadata features (if any)
            item_metadata_dim: Dimension of item metadata features (if any)
            temperature: Temperature for scaling similarity scores
            device: Computation device
        """
        super(DualEncoderModel, self).__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        
        # Text encoders
        self.user_text_encoder = user_text_encoder
        self.item_text_encoder = item_text_encoder or user_text_encoder
        
        # Get embedding dimensions
        self.user_embedding_dim = user_text_encoder.embedding_dim
        self.item_embedding_dim = self.item_text_encoder.embedding_dim
        
        # Determine if we need to handle metadata
        self.has_user_metadata = user_metadata_dim is not None and user_metadata_dim > 0
        self.has_item_metadata = item_metadata_dim is not None and item_metadata_dim > 0
        
        # Metadata encoders
        if self.has_user_metadata:
            self.user_metadata_encoder = Sequential(
                Linear(user_metadata_dim, self.user_embedding_dim),
                ReLU(),
                Dropout(0.2)
            )
        
        if self.has_item_metadata:
            self.item_metadata_encoder = Sequential(
                Linear(item_metadata_dim, self.item_embedding_dim),
                ReLU(),
                Dropout(0.2)
            )
        
        # Projections (if needed)
        self.projection_dim = projection_dim
        if projection_dim is not None:
            self.user_projection = Linear(self.user_embedding_dim, projection_dim)
            self.item_projection = Linear(self.item_embedding_dim, projection_dim)
    
    def encode_users(
        self, 
        user_texts: List[str],
        user_metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user texts and optional metadata into embeddings.
        
        Args:
            user_texts: List of user text descriptions
            user_metadata: Optional tensor of user metadata features
            
        Returns:
            User embeddings
        """
        # Encode user texts
        user_embeddings = self.user_text_encoder(user_texts)
        
        # Incorporate metadata if available
        if self.has_user_metadata and user_metadata is not None:
            metadata_embeddings = self.user_metadata_encoder(user_metadata)
            user_embeddings = user_embeddings + metadata_embeddings
        
        # Apply projection if needed
        if self.projection_dim is not None:
            user_embeddings = self.user_projection(user_embeddings)
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        
        return user_embeddings
    
    def encode_items(
        self, 
        item_texts: List[str],
        item_metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode item texts and optional metadata into embeddings.
        
        Args:
            item_texts: List of item text descriptions
            item_metadata: Optional tensor of item metadata features
            
        Returns:
            Item embeddings
        """
        # Encode item texts
        item_embeddings = self.item_text_encoder(item_texts)
        
        # Incorporate metadata if available
        if self.has_item_metadata and item_metadata is not None:
            metadata_embeddings = self.item_metadata_encoder(item_metadata)
            item_embeddings = item_embeddings + metadata_embeddings
        
        # Apply projection if needed
        if self.projection_dim is not None:
            item_embeddings = self.item_projection(item_embeddings)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        
        return item_embeddings
    
    def forward(
        self,
        user_texts: List[str],
        item_texts: List[str],
        user_metadata: Optional[torch.Tensor] = None,
        item_metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity scores between users and items.
        
        Args:
            user_texts: List of user text descriptions
            item_texts: List of item text descriptions
            user_metadata: Optional tensor of user metadata features
            item_metadata: Optional tensor of item metadata features
            
        Returns:
            Similarity scores
        """
        # Encode users and items
        user_embeddings = self.encode_users(user_texts, user_metadata)
        item_embeddings = self.encode_items(item_texts, item_metadata)
        
        # Compute similarity scores
        scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        scores = scores / self.temperature
        
        return scores
    
    def predict_pairs(
        self,
        user_texts: List[str],
        item_texts: List[str],
        user_metadata: Optional[torch.Tensor] = None,
        item_metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict scores for specific user-item pairs.
        
        Args:
            user_texts: List of user text descriptions
            item_texts: List of item text descriptions (same length as user_texts)
            user_metadata: Optional tensor of user metadata features
            item_metadata: Optional tensor of item metadata features
            
        Returns:
            Prediction scores for each pair
        """
        if len(user_texts) != len(item_texts):
            raise ValueError("user_texts and item_texts must have the same length")
        
        # Encode users and items
        user_embeddings = self.encode_users(user_texts, user_metadata)
        item_embeddings = self.encode_items(item_texts, item_metadata)
        
        # Compute dot product for each pair
        scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        scores = scores / self.temperature
        
        return scores


class HybridTextGNNModel(nn.Module):
    """
    Hybrid model that combines text embeddings with graph neural networks.
    """
    def __init__(
        self,
        text_encoder: TextEmbeddingEncoder,
        gnn_model: nn.Module,
        text_weight: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the hybrid model.
        
        Args:
            text_encoder: Text encoder model
            gnn_model: Graph neural network model
            text_weight: Weight for text embeddings (1-text_weight for GNN)
            device: Computation device
        """
        super(HybridTextGNNModel, self).__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = text_encoder
        self.gnn_model = gnn_model
        self.text_weight = text_weight
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: torch.Tensor,
        user_texts: List[str],
        item_texts: List[str]
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_label_index: Edge indices to predict
            user_texts: List of user text descriptions
            item_texts: List of item text descriptions
            
        Returns:
            Prediction scores
        """
        # Get GNN predictions
        gnn_scores = self.gnn_model(x_dict, edge_index_dict, edge_label_index)
        
        # Extract user and item indices from edge_label_index
        user_indices, item_indices = edge_label_index
        
        # Get corresponding texts
        batch_user_texts = [user_texts[i] for i in user_indices.cpu().numpy()]
        batch_item_texts = [item_texts[i] for i in item_indices.cpu().numpy()]
        
        # Get text-based predictions
        user_embeddings = self.text_encoder(batch_user_texts)
        item_embeddings = self.text_encoder(batch_item_texts)
        text_scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        # Normalize scores to similar ranges (optional)
        gnn_scores = torch.sigmoid(gnn_scores)
        text_scores = torch.sigmoid(text_scores)
        
        # Combine predictions
        combined_scores = self.text_weight * text_scores + (1 - self.text_weight) * gnn_scores
        
        return combined_scores
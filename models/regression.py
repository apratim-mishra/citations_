"""
Node regression models for predicting paper metrics such as Relative Citation Ratio.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple


class NodeRegressor(nn.Module):
    """
    Regression module for node-level prediction of continuous attributes.
    Takes embeddings from a GNN and predicts continuous values.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        """
        Initialize the node regressor.
        
        Args:
            in_channels: Input feature dimensionality
            hidden_channels: Hidden layer dimensionality
            num_layers: Number of hidden layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            batch_norm: Whether to use batch normalization
        """
        super(NodeRegressor, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Define activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Input layer
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Batch normalization layers
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            for i in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer (predicts a single continuous value)
        self.output_layer = nn.Linear(hidden_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of node regressor.
        
        Args:
            x: Node features or embeddings [num_nodes, in_channels]
            
        Returns:
            Predicted continuous values [num_nodes, 1]
        """
        # Input layer
        x = self.input_layer(x)
        if self.batch_norm:
            x = self.batch_norms[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            if self.batch_norm:
                x = self.batch_norms[i+1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output_layer(x)
        
        return x.squeeze(-1)  # Remove last dimension to get [num_nodes]


class MultiTaskGNN(nn.Module):
    """
    Multi-task GNN model for both link prediction and node regression.
    """
    def __init__(
        self,
        gnn_model: nn.Module,
        node_type: str = 'paper',
        regression_hidden_channels: int = 64,
        regression_num_layers: int = 2,
        regression_dropout: float = 0.2,
        mtl_weight: float = 0.5,  # Weight for balancing tasks (0-1)
        device: Optional[torch.device] = None
    ):
        """
        Initialize the multi-task GNN model.
        
        Args:
            gnn_model: Base GNN model for generating node embeddings
            node_type: Type of nodes to perform regression on ('paper')
            regression_hidden_channels: Hidden channels for regression model
            regression_num_layers: Number of hidden layers for regression model
            regression_dropout: Dropout rate for regression model
            mtl_weight: Weight for balancing link prediction and regression
                        (0 = only regression, 1 = only link prediction)
            device: Computation device
        """
        super(MultiTaskGNN, self).__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model = gnn_model
        self.node_type = node_type
        self.mtl_weight = mtl_weight
        
        # Determine the embedding dimension from the GNN model
        # Assuming the GNN model has a decoder with a user_transform module
        try:
            if hasattr(self.gnn_model, 'decoder') and hasattr(self.gnn_model.decoder, 'user_transform'):
                # Get the output dimension of the user_transform module
                embedding_dim = self.gnn_model.decoder.user_transform[-1].out_features
            else:
                # Default embedding dimension
                embedding_dim = 64
                print(f"Warning: Could not determine embedding dimension, using default: {embedding_dim}")
        except Exception as e:
            embedding_dim = 64
            print(f"Warning: Error determining embedding dimension: {e}. Using default: {embedding_dim}")
        
        # Create regressor module
        self.regressor = NodeRegressor(
            in_channels=embedding_dim,
            hidden_channels=regression_hidden_channels,
            num_layers=regression_num_layers,
            dropout=regression_dropout
        )
        
        # Move model to device
        self.to(self.device)
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: Optional[torch.Tensor] = None,
        mode: str = 'both'  # 'link', 'regression', or 'both'
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the multi-task model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_label_index: Edge indices for link prediction (optional)
            mode: Prediction mode ('link', 'regression', or 'both')
            
        Returns:
            Depending on mode:
            - 'link': Link prediction scores
            - 'regression': Regression predictions
            - 'both': Tuple of (link_pred, regression_pred)
        """
        # Get GNN embeddings for all nodes
        if hasattr(self.gnn_model, 'encoder'):
            # If the GNN model has a separate encoder
            node_embeddings = self.gnn_model.encoder(x_dict, edge_index_dict)
        else:
            # Otherwise, run a full forward pass and get embeddings from the model
            # This is a simplified approach and might need adaptation for specific GNN architectures
            with torch.no_grad():
                # Dummy edge_label_index if not provided
                dummy_edge_label = edge_label_index if edge_label_index is not None else torch.zeros((2, 1), device=self.device)
                self.gnn_model(x_dict, edge_index_dict, dummy_edge_label)
                # Access node embeddings (model-specific)
                # This is a placeholder and should be adapted to the actual GNN model
                if hasattr(self.gnn_model, 'node_embeddings'):
                    node_embeddings = self.gnn_model.node_embeddings
                else:
                    # Return an error if node embeddings cannot be accessed
                    raise AttributeError("Cannot access node embeddings from the GNN model. Please modify the implementation to match your GNN architecture.")
        
        # Link prediction
        if mode in ['link', 'both'] and edge_label_index is not None:
            link_pred = self.gnn_model(x_dict, edge_index_dict, edge_label_index)
        else:
            link_pred = None
        
        # Node regression
        if mode in ['regression', 'both']:
            # Get paper node embeddings
            if isinstance(node_embeddings, dict):
                # For heterogeneous GNNs
                paper_embeddings = node_embeddings[self.node_type]
            else:
                # For homogeneous GNNs
                paper_embeddings = node_embeddings
            
            # Run regression
            regression_pred = self.regressor(paper_embeddings)
        else:
            regression_pred = None
        
        # Return based on mode
        if mode == 'link':
            return link_pred
        elif mode == 'regression':
            return regression_pred
        else:  # both
            return link_pred, regression_pred
    
    def predict_rcr(
        self, 
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict Relative Citation Ratio for papers.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Predicted RCR values for all papers
        """
        return self.forward(x_dict, edge_index_dict, mode='regression')


class RCRPredictionGNN(nn.Module):
    """
    Specialized GNN for predicting Relative Citation Ratio of papers.
    This model focuses solely on the regression task.
    """
    def __init__(
        self,
        gnn_model: nn.Module,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        target_node_type: str = 'paper',
        device: Optional[torch.device] = None
    ):
        """
        Initialize RCR prediction model.
        
        Args:
            gnn_model: Base GNN model for node embeddings
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers for regression
            num_layers: Number of regression layers
            dropout: Dropout rate
            target_node_type: Node type to predict ('paper')
            device: Computation device
        """
        super(RCRPredictionGNN, self).__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model = gnn_model
        self.target_node_type = target_node_type
        
        # Define regression model
        self.regression_model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for RCR prediction.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Predicted RCR values
        """
        # Generate node embeddings
        with torch.no_grad():
            node_embeddings = self.gnn_model.encoder(x_dict, edge_index_dict)
        
        # Get paper embeddings
        paper_embeddings = node_embeddings[self.target_node_type]
        
        # Predict RCR
        predictions = self.regression_model(paper_embeddings)
        
        return predictions.squeeze(-1)  # Return [num_papers]
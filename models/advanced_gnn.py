"""
Advanced GNN models for recommendation systems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict, Sequential, Dropout, LayerNorm, ReLU
from torch_geometric.nn import (
    GATConv, 
    GCNConv, 
    SAGEConv, 
    TransformerConv, 
    HGTConv,
    to_hetero,
    HeteroConv
)
from typing import Dict, List, Optional, Tuple, Union, Any


class GNNLayerWithResidual(nn.Module):
    """
    GNN layer with residual connection, layer normalization, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = 'sage',
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        heads: int = 1,
        **kwargs
    ):
        """
        Initialize a GNN layer with residual connections.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            conv_type: Type of GNN convolution ('gcn', 'sage', 'gat', 'transformer')
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            heads: Number of attention heads (for GAT and Transformer)
            **kwargs: Additional arguments for the convolution layer
        """
        super(GNNLayerWithResidual, self).__init__()
        
        # Select the convolution type
        self.conv_type = conv_type
        self.use_layer_norm = use_layer_norm
        self.heads = heads
        
        if conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels, **kwargs)
        elif conv_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels, **kwargs)
        elif conv_type == 'gat':
            # GAT outputs head * out_channels features, so adjust output size
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, **kwargs)
        elif conv_type == 'transformer':
            # Transformer outputs head * out_channels features, so adjust output size
            self.conv = TransformerConv(in_channels, out_channels // heads, heads=heads, **kwargs)
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = LayerNorm(out_channels)
        
        # Projection for residual connection if dimensions don't match
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = Linear(in_channels, out_channels)
        
        # Dropout
        self.dropout = Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features (optional)
            
        Returns:
            Updated node features
        """
        # Apply convolution
        if edge_attr is not None and self.conv_type in ['gcn', 'gat', 'transformer']:
            # For convolutions that support edge features
            conv_out = self.conv(x, edge_index, edge_attr)
        else:
            conv_out = self.conv(x, edge_index)
        
        # Apply residual connection
        if self.use_projection:
            identity = self.projection(x)
        else:
            identity = x
        
        # Combine with residual and apply activation
        out = conv_out + identity
        out = F.relu(out)
        
        # Apply layer normalization
        if self.use_layer_norm:
            out = self.layer_norm(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out


class AdvancedGNNEncoder(nn.Module):
    """
    Advanced GNN encoder with multiple layers, skip connections, and heterogeneous support.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        conv_type: str = 'sage',
        dropout: float = 0.2,
        residual: bool = True,
        use_layer_norm: bool = True,
        attention_heads: int = 4,
        skip_connections: bool = True
    ):
        """
        Initialize the GNN encoder.
        
        Args:
            in_channels_dict: Dictionary mapping node types to input feature dimensions
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output embeddings
            num_layers: Number of GNN layers
            conv_type: Type of GNN convolution
            dropout: Dropout probability
            residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
            attention_heads: Number of attention heads for GAT and Transformer
            skip_connections: Whether to use skip connections from all layers to output
        """
        super(AdvancedGNNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        # Initial projection for each node type
        self.node_type_encoders = ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.node_type_encoders[node_type] = Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Dropout(dropout)
            )
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        # For each layer, create a heterogeneous convolution
        for layer in range(num_layers):
            # Input channels for this layer
            in_size = hidden_channels
            # Output channels for this layer (last layer has out_channels)
            out_size = out_channels if layer == num_layers - 1 else hidden_channels
            
            # Create a GNN layer with the specified type
            layer_with_residual = lambda in_c, out_c: GNNLayerWithResidual(
                in_c, out_c, 
                conv_type=conv_type,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                heads=attention_heads
            )
            
            # Add the layer
            if layer == 0:
                # First layer might need special handling
                self.convs.append(layer_with_residual(in_size, out_size))
            else:
                self.convs.append(layer_with_residual(in_size, out_size))
        
        # Skip connection projections (if dimensions differ)
        if skip_connections and hidden_channels != out_channels:
            self.skip_projections = nn.ModuleList([
                Linear(hidden_channels, out_channels) for _ in range(num_layers - 1)
            ])
        else:
            self.skip_projections = None
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary of node embeddings
        """
        # Initial projection of node features
        hidden_dict = {}
        for node_type, x in x_dict.items():
            hidden_dict[node_type] = self.node_type_encoders[node_type](x)
        
        # For storing intermediate representations for skip connections
        intermediate_reps = [] if self.skip_connections else None
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            # Store the current representation for skip connections
            if self.skip_connections and i < self.num_layers - 1:
                intermediate_reps.append(hidden_dict.copy())
            
            # Forward through the convolution (with to_hetero wrapping)
            # Note: This is a simplified representation - in a real implementation,
            # you would apply the convolution to each edge type and update the nodes
            conv_het = to_hetero(conv, edge_index_dict, aggr='sum')
            hidden_dict = conv_het(hidden_dict, edge_index_dict)
        
        # Apply skip connections if enabled
        if self.skip_connections:
            for i, intermediate_dict in enumerate(intermediate_reps):
                for node_type in hidden_dict.keys():
                    # Project if dimensions differ
                    if self.skip_projections is not None:
                        projected = self.skip_projections[i](intermediate_dict[node_type])
                        hidden_dict[node_type] = hidden_dict[node_type] + projected
                    else:
                        hidden_dict[node_type] = hidden_dict[node_type] + intermediate_dict[node_type]
        
        return hidden_dict


class HGTEncoder(nn.Module):
    """
    Hierarchical Graph Transformer (HGT) encoder for heterogeneous graphs.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        metadata=None  # (node_types, edge_types)
    ):
        """
        Initialize the HGT encoder.
        
        Args:
            in_channels_dict: Dictionary mapping node types to input feature dimensions
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output embeddings
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            metadata: Graph metadata (node_types, edge_types)
        """
        super(HGTEncoder, self).__init__()
        
        if metadata is None:
            raise ValueError("Metadata is required for HGT encoder")
            
        # Unpack metadata
        node_types, edge_types = metadata
        
        # Linear transformations for input embedding
        self.lin_dict = ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(in_channels_dict[node_type], hidden_channels)
        
        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout
            )
            self.convs.append(conv)
        
        # Output transformation
        self.out_dict = ModuleDict()
        for node_type in node_types:
            self.out_dict[node_type] = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary of node embeddings
        """
        # Initial linear transformation
        h_dict = {node_type: F.relu(self.lin_dict[node_type](x)) 
                 for node_type, x in x_dict.items()}
        
        # Apply HGT convolutions
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply non-linearity
            h_dict = {node_type: F.relu(h) for node_type, h in h_dict.items()}
        
        # Apply output transformation
        out_dict = {node_type: self.out_dict[node_type](h) 
                   for node_type, h in h_dict.items()}
        
        return out_dict


class CustomHeteroGNN(nn.Module):
    """
    Custom heterogeneous GNN that applies different convolutions to different edge types.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        metadata=None,  # (node_types, edge_types)
        dropout: float = 0.2,
        residual: bool = True
    ):
        """
        Initialize the custom heterogeneous GNN.
        
        Args:
            in_channels_dict: Dictionary mapping node types to input feature dimensions
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output embeddings
            num_layers: Number of GNN layers
            metadata: Graph metadata (node_types, edge_types)
            dropout: Dropout probability
            residual: Whether to use residual connections
        """
        super(CustomHeteroGNN, self).__init__()
        
        if metadata is None:
            raise ValueError("Metadata is required for custom heterogeneous GNN")
        
        # Node embedding layers
        self.embeddings = ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.embeddings[node_type] = Linear(in_channels, hidden_channels)
        
        # Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Create a dict of convs for each edge type
            conv_dict = {}
            
            for edge_type in metadata[1]:
                # Use different convolution types based on edge semantics
                src_type, edge_name, dst_type = edge_type
                
                if edge_name == 'writes' or edge_name == 'authored':
                    # Use GAT for user-item relationships
                    conv_dict[edge_type] = GATConv(
                        (hidden_channels, hidden_channels), 
                        hidden_channels // 4,
                        heads=4,
                        dropout=dropout
                    )
                elif edge_name == 'similar_to' or edge_name == 'cites':
                    # Use SAGE for item-item relationships
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels
                    )
                else:
                    # Default to GCN for other relationships
                    conv_dict[edge_type] = GCNConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels
                    )
            
            # Create a heterogeneous convolution layer
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)
        
        # Layer normalization and dropout
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = {}
            for node_type in metadata[0]:
                norm_dict[node_type] = LayerNorm(hidden_channels)
            self.norms.append(ModuleDict(norm_dict))
        
        # Output layers
        self.outputs = ModuleDict()
        for node_type in metadata[0]:
            self.outputs[node_type] = Linear(hidden_channels, out_channels)
        
        # Dropout
        self.dropout = dropout
        self.residual = residual
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary of node embeddings
        """
        # Initial node embedding
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.relu(self.embeddings[node_type](x))
        
        # Apply GNN layers
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Store previous representations for residual connections
            prev_h_dict = h_dict.copy() if self.residual else None
            
            # Apply convolution
            h_dict = conv(h_dict, edge_index_dict)
            
            # Apply layer normalization and non-linearity
            for node_type in h_dict.keys():
                h_dict[node_type] = norm_dict[node_type](h_dict[node_type])
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = F.dropout(h_dict[node_type], p=self.dropout, training=self.training)
                
                # Apply residual connection
                if self.residual and node_type in prev_h_dict:
                    h_dict[node_type] = h_dict[node_type] + prev_h_dict[node_type]
        
        # Apply output transformation
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.outputs[node_type](h)
        
        return out_dict


class AdvancedGNNModel(torch.nn.Module):
    """
    Advanced GNN model for recommendation with various encoder options.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        encoder_type: str = 'advanced',
        num_layers: int = 4,
        dropout: float = 0.2,
        residual: bool = True,
        layer_norm: bool = True,
        attention_heads: int = 4,
        skip_connections: bool = True,
        metadata=None,
        device=None
    ):
        """
        Initialize the model.
        
        Args:
            in_channels_dict: Dictionary mapping node types to input feature dimensions
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output embeddings
            encoder_type: Type of encoder ('advanced', 'hgt', 'custom')
            num_layers: Number of layers in the encoder
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to use layer normalization
            attention_heads: Number of attention heads for attention-based models
            skip_connections: Whether to use skip connections
            metadata: Graph metadata for heterogeneous models
            device: Computation device
        """
        super(AdvancedGNNModel, self).__init__()
        
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Select encoder
        if encoder_type == 'advanced':
            self.encoder = AdvancedGNNEncoder(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                residual=residual,
                use_layer_norm=layer_norm,
                attention_heads=attention_heads,
                skip_connections=skip_connections
            )
            # Convert to heterogeneous
            self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        elif encoder_type == 'hgt':
            self.encoder = HGTEncoder(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                num_heads=attention_heads,
                dropout=dropout,
                metadata=metadata
            )
        elif encoder_type == 'custom':
            self.encoder = CustomHeteroGNN(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                metadata=metadata,
                dropout=dropout,
                residual=residual
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Edge decoder for link prediction
        self.decoder = EdgeDecoder(out_channels)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_label_index: Edge indices to predict
            
        Returns:
            Prediction scores for the specified edges
        """
        # Generate node embeddings
        node_embeddings = self.encoder(x_dict, edge_index_dict)
        
        # Use decoder to predict links
        return self.decoder(node_embeddings, edge_label_index)


class EdgeDecoder(nn.Module):
    """
    Edge decoder for link prediction that supports multiple user and item types.
    """
    def __init__(self, hidden_channels: int):
        """
        Initialize the edge decoder.
        
        Args:
            hidden_channels: Dimension of node embeddings
        """
        super(EdgeDecoder, self).__init__()
        
        # Transformations for each node type
        self.user_transform = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels)
        )
        
        self.item_transform = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, z_dict, edge_label_index):
        """
        Forward pass.
        
        Args:
            z_dict: Dictionary of node embeddings
            edge_label_index: Edge indices to predict (source, target)
            
        Returns:
            Prediction scores
        """
        row, col = edge_label_index
        
        # Determine node types from edge_label_index
        # This assumes edge_label_index refers to a specific edge type (e.g., user-item)
        # In a real scenario, you might need to explicitly specify the node types
        user_type = 'user'  # or determine dynamically
        item_type = 'restaurant'  # or determine dynamically
        
        # Get embeddings
        user_embeddings = z_dict[user_type][row]
        item_embeddings = z_dict[item_type][col]
        
        # Transform embeddings
        user_transformed = self.user_transform(user_embeddings)
        item_transformed = self.item_transform(item_embeddings)
        
        # Calculate dot product
        return (user_transformed * item_transformed).sum(dim=-1)
"""
Training pipeline for multi-task learning with link prediction and node regression.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import wandb

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, LinkNeighborLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.baseline import BaselineModel
from models.improved import ImprovedModel
from models.advanced_gnn import AdvancedGNNModel
from models.regression import MultiTaskGNN, RCRPredictionGNN, NodeRegressor

from utils.preprocessing import split_edge_data, split_node_data, load_tensor, save_tensor
from utils.evaluation import validate, compute_metrics
from utils.logger import setup_logger
from utils.model_utils import count_parameters

import config.model_config as cfg

# Setup logger
logger = setup_logger()


def train_multitask(
    graph: HeteroData,
    rcr_key: str = 'relative_citation_ratio',
    base_model_type: str = 'improved',
    model_config: Optional[Dict[str, Any]] = None,
    mtl_weight: float = 0.5,  # Weight between link prediction (1.0) and regression (0.0)
    num_epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    val_frequency: int = 5,
    k: int = 10,
    save_dir: str = 'trained_models',
    device: Optional[torch.device] = None,
    use_wandb: bool = False
) -> Tuple[MultiTaskGNN, Dict[str, Any]]:
    """
    Train a multi-task model for link prediction and RCR regression.
    
    Args:
        graph: HeteroData graph
        rcr_key: Key for the RCR value in graph['paper']
        base_model_type: Base GNN model type ('baseline', 'improved', 'advanced_gnn')
        model_config: Model configuration
        mtl_weight: Weight between link prediction and regression tasks (0-1)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for L2 regularization
        early_stopping_patience: Patience for early stopping
        val_frequency: Validation frequency in epochs
        k: K value for link prediction metrics
        save_dir: Directory to save model checkpoints
        device: Computation device
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Tuple of (trained model, results dictionary)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project="citation-prediction",
            config={
                "model_type": f"multitask_{base_model_type}",
                "mtl_weight": mtl_weight,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        )
    
    # Check if graph has the RCR attribute
    if rcr_key not in graph['paper'].keys:
        raise ValueError(f"Graph does not have '{rcr_key}' attribute for paper nodes")
    
    # Load config if not provided
    if model_config is None:
        config = cfg.load_config()
        if base_model_type in config['model']:
            model_config = config['model'][base_model_type]
        else:
            model_config = config['model']['improved']  # Default
    
    # Prepare data splits for link prediction
    logger.info("Splitting edge data for link prediction...")
    edge_train_data, edge_val_data, edge_test_data = split_edge_data(
        graph, 'author', 'paper', edge_type=('paper', 'cites', 'paper')
    )
    
    # Prepare data splits for node regression
    logger.info("Splitting node data for regression...")
    node_mask_train, node_mask_val, node_mask_test = split_node_data(
        graph, 'paper', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Get RCR values
    rcr_values = graph['paper'][rcr_key].float()
    
    # Create base GNN model
    logger.info(f"Creating base {base_model_type} GNN model...")
    metadata = (graph.node_types, graph.edge_types)
    
    if base_model_type == 'baseline':
        base_model = BaselineModel(
            metadata=metadata,
            hidden_channels=model_config.get('hidden_channels', 64),
            device=device
        )
    elif base_model_type == 'improved':
        base_model = ImprovedModel(
            metadata=metadata,
            hidden_channels=model_config.get('hidden_channels', 64),
            device=device
        )
    elif base_model_type == 'advanced_gnn':
        # Get feature dimensions for each node type
        in_channels_dict = {node_type: graph[node_type].x.size(1) for node_type in graph.node_types}
        
        base_model = AdvancedGNNModel(
            in_channels_dict=in_channels_dict,
            hidden_channels=model_config.get('hidden_channels', 128),
            out_channels=model_config.get('out_channels', 64),
            encoder_type='advanced',
            num_layers=model_config.get('num_layers', 4),
            dropout=model_config.get('dropout', 0.2),
            metadata=metadata,
            device=device
        )
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")
    
    # Create multi-task model
    logger.info("Creating multi-task model...")
    model = MultiTaskGNN(
        gnn_model=base_model,
        node_type='paper',
        regression_hidden_channels=model_config.get('hidden_channels', 64),
        regression_num_layers=model_config.get('num_layers', 2),
        regression_dropout=model_config.get('dropout', 0.2),
        mtl_weight=mtl_weight,
        device=device
    )
    
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Setup data loader for link prediction
    train_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors=[10, 5],  # Number of neighbors to sample for each node
        edge_label_index=(('paper', 'cites', 'paper'), edge_train_data[0]),
        edge_label=edge_train_data[1],
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting multi-task training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    results = {
        'train_link_loss': [],
        'train_regression_loss': [],
        'train_total_loss': [],
        'val_link_metrics': [],
        'val_regression_metrics': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'training_time': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_link_loss = 0.0
        total_regression_loss = 0.0
        total_combined_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Get edge label index for citation edges
            edge_label_index = batch[('paper', 'cites', 'paper')].edge_label_index
            edge_label = batch[('paper', 'cites', 'paper')].edge_label
            
            # Get paper node indices and RCR values for this batch
            paper_indices = torch.unique(batch['paper'].n_id)
            batch_rcr = rcr_values[paper_indices].to(device)
            
            # Get training mask for papers in this batch
            batch_train_mask = node_mask_train[paper_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - multi-task
            link_pred, regression_pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                edge_label_index,
                mode='both'
            )
            
            # Calculate link prediction loss
            link_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                link_pred, edge_label.float()
            )
            
            # Calculate regression loss (only for training nodes)
            if batch_train_mask.sum() > 0:
                regression_loss = torch.nn.functional.mse_loss(
                    regression_pred[batch_train_mask],
                    batch_rcr[batch_train_mask]
                )
            else:
                regression_loss = torch.tensor(0.0, device=device)
            
            # Combined loss with task weighting
            combined_loss = mtl_weight * link_loss + (1 - mtl_weight) * regression_loss
            
            # Backward pass
            combined_loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_link_loss += link_loss.item()
            total_regression_loss += regression_loss.item()
            total_combined_loss += combined_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'link_loss': link_loss.item(),
                'reg_loss': regression_loss.item(),
                'total_loss': combined_loss.item()
            })
        
        # Calculate average losses for the epoch
        avg_link_loss = total_link_loss / max(1, num_batches)
        avg_regression_loss = total_regression_loss / max(1, num_batches)
        avg_combined_loss = total_combined_loss / max(1, num_batches)
        
        results['train_link_loss'].append(avg_link_loss)
        results['train_regression_loss'].append(avg_regression_loss)
        results['train_total_loss'].append(avg_combined_loss)
        
        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                "train_link_loss": avg_link_loss,
                "train_regression_loss": avg_regression_loss,
                "train_total_loss": avg_combined_loss,
                "epoch": epoch
            })
        
        # Validation
        if epoch % val_frequency == 0 or epoch == num_epochs:
            logger.info(f"Evaluating model at epoch {epoch}...")
            model.eval()
            
            # Link prediction validation
            link_metrics = validate(
                model, graph, edge_val_data,
                k_values=[k], device=device
            )
            
            # Regression validation
            val_paper_indices = torch.arange(graph['paper'].num_nodes)[node_mask_val]
            val_rcr = rcr_values[val_paper_indices].cpu().numpy()
            
            with torch.no_grad():
                # Forward pass for all papers
                val_pred_rcr = model.predict_rcr(
                    graph.x_dict,
                    graph.edge_index_dict
                )
                # Get predictions for validation papers
                val_pred_rcr = val_pred_rcr[val_paper_indices].cpu().numpy()
            
            # Calculate regression metrics
            reg_metrics = {
                'mse': mean_squared_error(val_rcr, val_pred_rcr),
                'mae': mean_absolute_error(val_rcr, val_pred_rcr),
                'r2': r2_score(val_rcr, val_pred_rcr)
            }
            
            # Store validation metrics
            results['val_link_metrics'].append(link_metrics)
            results['val_regression_metrics'].append(reg_metrics)
            
            # Calculate combined validation loss
            val_link_loss = 1 - link_metrics.get(f'ndcg@{k}', 0)  # Convert NDCG to loss
            val_reg_loss = reg_metrics['mse']
            val_combined_loss = mtl_weight * val_link_loss + (1 - mtl_weight) * val_reg_loss
            
            # Log to Weights & Biases
            if use_wandb:
                wandb.log({
                    "val_link_loss": val_link_loss,
                    "val_ndcg": link_metrics.get(f'ndcg@{k}', 0),
                    "val_mse": reg_metrics['mse'],
                    "val_mae": reg_metrics['mae'],
                    "val_r2": reg_metrics['r2'],
                    "val_combined_loss": val_combined_loss,
                    "epoch": epoch
                })
            
            # Update learning rate scheduler
            scheduler.step(val_combined_loss)
            
            # Check for best model and early stopping
            if val_combined_loss < best_val_loss:
                best_val_loss = val_combined_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(save_dir, f"multitask_{base_model_type}_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Print validation metrics
            logger.info(f"Epoch {epoch} | Link NDCG@{k}: {link_metrics.get(f'ndcg@{k}', 0):.4f} | "
                       f"Regression MSE: {reg_metrics['mse']:.4f}, R²: {reg_metrics['r2']:.4f}")
    
    # Calculate total training time
    training_time = time.time() - start_time
    results['training_time'] = training_time
    results['best_epoch'] = best_epoch
    results['best_val_loss'] = best_val_loss
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    model_path = os.path.join(save_dir, f"multitask_{base_model_type}_best.pt")
    model.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded best model from {model_path}")
    
    # Save final model if different from best
    final_model_path = os.path.join(save_dir, f"multitask_{base_model_type}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Finalize Weights & Biases
    if use_wandb:
        wandb.finish()
    
    return model, results


def train_regression_only(
    graph: HeteroData,
    rcr_key: str = 'relative_citation_ratio',
    base_model_type: str = 'improved',
    model_config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    val_frequency: int = 5,
    save_dir: str = 'trained_models',
    device: Optional[torch.device] = None,
    use_wandb: bool = False
) -> Tuple[RCRPredictionGNN, Dict[str, Any]]:
    """
    Train a model for RCR regression only.
    
    Args:
        graph: HeteroData graph
        rcr_key: Key for the RCR value in graph['paper']
        base_model_type: Base GNN model type ('baseline', 'improved', 'advanced_gnn')
        model_config: Model configuration
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for L2 regularization
        early_stopping_patience: Patience for early stopping
        val_frequency: Validation frequency in epochs
        save_dir: Directory to save model checkpoints
        device: Computation device
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Tuple of (trained model, results dictionary)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project="citation-prediction",
            config={
                "model_type": f"regression_{base_model_type}",
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        )
    
    # Check if graph has the RCR attribute
    if rcr_key not in graph['paper'].keys:
        raise ValueError(f"Graph does not have '{rcr_key}' attribute for paper nodes")
    
    # Load config if not provided
    if model_config is None:
        config = cfg.load_config()
        if base_model_type in config['model']:
            model_config = config['model'][base_model_type]
        else:
            model_config = config['model']['improved']  # Default
    
    # Prepare data splits for node regression
    logger.info("Splitting node data for regression...")
    node_mask_train, node_mask_val, node_mask_test = split_node_data(
        graph, 'paper', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Get RCR values
    rcr_values = graph['paper'][rcr_key].float()
    
    # Create base GNN model
    logger.info(f"Creating base {base_model_type} GNN model...")
    metadata = (graph.node_types, graph.edge_types)
    
    if base_model_type == 'baseline':
        base_model = BaselineModel(
            metadata=metadata,
            hidden_channels=model_config.get('hidden_channels', 64),
            device=device
        )
    elif base_model_type == 'improved':
        base_model = ImprovedModel(
            metadata=metadata,
            hidden_channels=model_config.get('hidden_channels', 64),
            device=device
        )
    elif base_model_type == 'advanced_gnn':
        # Get feature dimensions for each node type
        in_channels_dict = {node_type: graph[node_type].x.size(1) for node_type in graph.node_types}
        
        base_model = AdvancedGNNModel(
            in_channels_dict=in_channels_dict,
            hidden_channels=model_config.get('hidden_channels', 128),
            out_channels=model_config.get('out_channels', 64),
            encoder_type='advanced',
            num_layers=model_config.get('num_layers', 4),
            dropout=model_config.get('dropout', 0.2),
            metadata=metadata,
            device=device
        )
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")
    
    # Create regression model
    logger.info("Creating RCR prediction model...")
    model = RCRPredictionGNN(
        gnn_model=base_model,
        embedding_dim=model_config.get('out_channels', 64),
        hidden_dim=model_config.get('hidden_channels', 128),
        num_layers=model_config.get('num_layers', 3),
        dropout=model_config.get('dropout', 0.3),
        target_node_type='paper',
        device=device
    )
    
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create train/val/test node indices
    train_indices = torch.arange(graph['paper'].num_nodes)[node_mask_train]
    val_indices = torch.arange(graph['paper'].num_nodes)[node_mask_val]
    
    # Create batch sampler for training
    train_loader = torch.utils.data.DataLoader(
        train_indices.tolist(),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting regression training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    results = {
        'train_loss': [],
        'val_metrics': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'training_time': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_indices in progress_bar:
            # Convert batch indices to tensor and move to device
            batch_indices = torch.tensor(batch_indices, device=device)
            
            # Get RCR values for this batch
            batch_rcr = rcr_values[batch_indices].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - predict RCR for all papers
            all_predictions = model(graph.x_dict, graph.edge_index_dict)
            
            # Extract predictions for the batch
            batch_predictions = all_predictions[batch_indices]
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(batch_predictions, batch_rcr)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / max(1, num_batches)
        results['train_loss'].append(avg_loss)
        
        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                "train_loss": avg_loss,
                "epoch": epoch
            })
        
        # Validation
        if epoch % val_frequency == 0 or epoch == num_epochs:
            logger.info(f"Evaluating model at epoch {epoch}...")
            model.eval()
            
            with torch.no_grad():
                # Predict RCR for all papers
                all_predictions = model(graph.x_dict, graph.edge_index_dict)
                
                # Get predictions for validation papers
                val_predictions = all_predictions[val_indices].cpu().numpy()
                val_true = rcr_values[val_indices].cpu().numpy()
                
                # Calculate validation metrics
                val_mse = mean_squared_error(val_true, val_predictions)
                val_mae = mean_absolute_error(val_true, val_predictions)
                val_r2 = r2_score(val_true, val_predictions)
            
            # Store validation metrics
            results['val_metrics'].append({
                'mse': val_mse,
                'mae': val_mae,
                'r2': val_r2
            })
            
            # Log to Weights & Biases
            if use_wandb:
                wandb.log({
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "epoch": epoch
                })
            
            # Update learning rate scheduler
            scheduler.step(val_mse)
            
            # Check for best model and early stopping
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(save_dir, f"regression_{base_model_type}_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Print validation metrics
            logger.info(f"Epoch {epoch} | Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Calculate total training time
    training_time = time.time() - start_time
    results['training_time'] = training_time
    results['best_epoch'] = best_epoch
    results['best_val_loss'] = best_val_loss
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation MSE: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    model_path = os.path.join(save_dir, f"regression_{base_model_type}_best.pt")
    model.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded best model from {model_path}")
    
    # Save final model if different from best
    final_model_path = os.path.join(save_dir, f"regression_{base_model_type}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Finalize Weights & Biases
    if use_wandb:
        wandb.finish()
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-task models for link prediction and node regression')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory with processed graph data')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--task', type=str, choices=['multitask', 'regression'], default='multitask',
                        help='Training task: multitask or regression-only')
    parser.add_argument('--base_model', type=str, default='improved',
                        help='Base GNN model type')
    parser.add_argument('--rcr_key', type=str, default='relative_citation_ratio',
                        help='Key for RCR values in graph')
    parser.add_argument('--mtl_weight', type=float, default=0.5,
                        help='Weight for multi-task learning (0 = only regression, 1 = only link prediction)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--k', type=int, default=10,
                        help='K value for link prediction metrics')
    parser.add_argument('--save_dir', type=str, default='trained_models',
                        help='Directory to save models')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases for tracking')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load graph data
    graph_path = os.path.join(args.data_dir, 'yelp_graph.pt')
    logger.info(f"Loading graph from {graph_path}...")
    graph = load_tensor(graph_path)
    
    # Load configuration
    config = cfg.load_config(args.config)
    
    if args.task == 'multitask':
        # Train multi-task model
        logger.info("Training multi-task model...")
        model, results = train_multitask(
            graph=graph,
            rcr_key=args.rcr_key,
            base_model_type=args.base_model,
            model_config=config['model'].get(args.base_model),
            mtl_weight=args.mtl_weight,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            k=args.k,
            save_dir=args.save_dir,
            device=device,
            use_wandb=args.wandb
        )
    else:
        # Train regression-only model
        logger.info("Training regression-only model...")
        model, results = train_regression_only(
            graph=graph,
            rcr_key=args.rcr_key,
            base_model_type=args.base_model,
            model_config=config['model'].get(args.base_model),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            device=device,
            use_wandb=args.wandb
        )
    
    logger.info("Training complete!")
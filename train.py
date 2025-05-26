"""
Training pipeline for recommendation models.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from tqdm import tqdm
import wandb

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, LinkNeighborLoader

from models.baseline import BaselineModel
from models.improved import ImprovedModel
from models.advanced_gnn import AdvancedGNNModel
from models.sentence_models import HybridTextGNNModel, TextEmbeddingEncoder

from losses.bpr_loss import BPRLoss, EnhancedBPRLoss
from losses.adaptive_loss import AdaptiveHybridLoss, FocalLoss

from utils.preprocessing import split_edge_data, load_tensor, save_tensor
from utils.evaluation import validate, compute_metrics
from utils.sampling import HardNegativeSampler
from utils.logger import setup_logger
from utils.model_utils import count_parameters

import config.model_config as cfg

# Setup logger
logger = setup_logger()


def get_model(
    model_type: str,
    graph: HeteroData,
    config: Dict[str, Any],
    device: torch.device
) -> torch.nn.Module:
    """
    Get the appropriate model based on model_type.
    
    Args:
        model_type: Type of model ('baseline', 'improved', 'advanced_gnn', 'hybrid')
        graph: HeteroData graph
        config: Model configuration
        device: Computation device
        
    Returns:
        Model instance
    """
    metadata = (graph.node_types, graph.edge_types)
    
    # Get feature dimensions for each node type
    in_channels_dict = {node_type: graph[node_type].x.size(1) for node_type in graph.node_types}
    
    if model_type == 'baseline':
        model_cfg = config['model']['baseline']
        model = BaselineModel(
            metadata=metadata,
            hidden_channels=model_cfg['hidden_channels'],
            device=device
        )
    elif model_type == 'improved':
        model_cfg = config['model']['improved']
        model = ImprovedModel(
            metadata=metadata,
            hidden_channels=model_cfg['hidden_channels'],
            device=device
        )
    elif model_type == 'advanced_gnn':
        model_cfg = config['model']['advanced_gnn']
        model = AdvancedGNNModel(
            in_channels_dict=in_channels_dict,
            hidden_channels=model_cfg['hidden_channels'],
            out_channels=model_cfg['out_channels'],
            encoder_type='advanced',
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            residual=model_cfg['residual'],
            layer_norm=model_cfg['layer_norm'],
            attention_heads=model_cfg['attention_heads'],
            skip_connections=True,
            metadata=metadata,
            device=device
        )
    elif model_type == 'hybrid':
        model_cfg = config['model']['hybrid']
        
        # Create text encoder
        sentence_cfg = config['model']['sentence']
        text_encoder = TextEmbeddingEncoder(
            model_name=sentence_cfg['model_name'],
            output_dim=model_cfg['text_hidden_dim'],
            pooling_mode=sentence_cfg['pooling_mode'],
            max_seq_length=sentence_cfg['max_seq_length'],
            device=device
        )
        
        # Create base GNN model
        gnn_model = ImprovedModel(
            metadata=metadata,
            hidden_channels=model_cfg['gnn_hidden_dim'],
            device=device
        )
        
        # Create hybrid model
        model = HybridTextGNNModel(
            text_encoder=text_encoder,
            gnn_model=gnn_model,
            text_weight=0.5,  # Equal weighting for text and graph components
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Log model parameters
    num_params = count_parameters(model)
    logger.info(f"Model: {model_type} with {num_params:,} parameters")
    
    return model


def get_loss_function(
    loss_type: str, 
    config: Dict[str, Any]
) -> torch.nn.Module:
    """
    Get the appropriate loss function.
    
    Args:
        loss_type: Type of loss function ('bce', 'bpr', 'contrastive', 'adaptive')
        config: Loss configuration
        
    Returns:
        Loss function module
    """
    loss_cfg = config['loss']
    
    if loss_type == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == 'bpr':
        return BPRLoss()
    elif loss_type == 'enhanced_bpr':
        return EnhancedBPRLoss(margin=0.5, user_reg=1e-6, item_reg=1e-6)
    elif loss_type == 'focal':
        focal_cfg = loss_cfg.get('adaptive', {})
        return FocalLoss(
            alpha=focal_cfg.get('alpha', 0.25),
            gamma=focal_cfg.get('gamma', 2.0)
        )
    elif loss_type == 'adaptive':
        adaptive_cfg = loss_cfg.get('adaptive', {})
        return AdaptiveHybridLoss(
            alpha=adaptive_cfg.get('alpha', 0.25),
            gamma=adaptive_cfg.get('gamma', 2.0),
            beta=adaptive_cfg.get('beta', 0.5)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def train_model(
    graph: HeteroData,
    model_type: str = 'improved',
    loss_type: str = 'bce',
    config: Optional[Dict[str, Any]] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    k: Optional[int] = None,
    hard_sampling: bool = False,
    ref_model: Optional[torch.nn.Module] = None,
    save_dir: str = 'trained_models',
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
    wandb_project: str = 'yelp-recommendation',
    wandb_entity: Optional[str] = None
) -> Tuple[torch.nn.Module, Tuple, Dict[str, Any]]:
    """
    Train a recommendation model.
    
    Args:
        graph: HeteroData graph
        model_type: Type of model ('baseline', 'improved', 'advanced_gnn', 'hybrid')
        loss_type: Type of loss function ('bce', 'bpr', 'enhanced_bpr', 'focal', 'adaptive')
        config: Model configuration (if None, loads from config module)
        num_epochs: Number of training epochs (overrides config)
        batch_size: Batch size for training (overrides config)
        lr: Learning rate (overrides config)
        k: Value of k for evaluation metrics (overrides config)
        hard_sampling: Whether to use hard negative sampling
        ref_model: Reference model for hard negative sampling
        save_dir: Directory to save model checkpoints
        device: Computation device
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        
    Returns:
        Tuple of (trained model, data splits, results dictionary)
    """
    # Load config if not provided
    if config is None:
        config = cfg.load_config()
    
    # Override config with function arguments if provided
    train_cfg = config['training']
    if num_epochs is not None:
        train_cfg['num_epochs'] = num_epochs
    if batch_size is not None:
        train_cfg['batch_size'] = batch_size
    if lr is not None:
        train_cfg['learning_rate'] = lr
    
    eval_cfg = config['evaluation']
    if k is not None:
        # Ensure k is in k_values
        if k not in eval_cfg['k_values']:
            eval_cfg['k_values'] = sorted(eval_cfg['k_values'] + [k])
    else:
        # Use the largest k value in the config
        k = max(eval_cfg['k_values'])
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "model_type": model_type,
                "loss_type": loss_type,
                "num_epochs": train_cfg['num_epochs'],
                "batch_size": train_cfg['batch_size'],
                "learning_rate": train_cfg['learning_rate'],
                "weight_decay": train_cfg['weight_decay'],
                "hard_sampling": hard_sampling,
                "device": str(device)
            }
        )
    
    # Prepare data splits
    logger.info("Splitting edge data...")
    train_data, val_data, test_data = split_edge_data(graph, 'user', 'restaurant')
    
    # Create model
    model = get_model(model_type, graph, config, device)
    
    # Get loss function
    criterion = get_loss_function(loss_type, config)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # Setup learning rate scheduler
    scheduler_type = train_cfg.get('scheduler', 'reduce_on_plateau')
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=train_cfg['scheduler_params'].get('factor', 0.5),
            patience=train_cfg['scheduler_params'].get('patience', 5),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['num_epochs'],
            eta_min=train_cfg['scheduler_params'].get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # Setup negative sampler if hard sampling is enabled
    if hard_sampling:
        if ref_model is None:
            logger.warning("Hard sampling enabled but no reference model provided. Using random sampling.")
            hard_sampler = None
        else:
            logger.info("Initializing hard negative sampler...")
            hard_sampler = HardNegativeSampler(
                graph=graph,
                ref_model=ref_model,
                device=device
            )
    else:
        hard_sampler = None
    
    # Setup gradient clipping
    grad_clip_value = train_cfg.get('gradient_clipping')
    
    # Setup data loader
    train_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors=[10, 5],  # Number of neighbors to sample for each node
        edge_label_index=(('user', 'rates', 'restaurant'), train_data[0]),
        edge_label=train_data[1],
        batch_size=train_cfg['batch_size'],
        shuffle=True
    )
    
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {train_cfg['num_epochs']} epochs...")
    
    best_val_score = 0.0
    best_epoch = 0
    results = {
        'train_loss': [],
        'val_metrics': [],
        'best_epoch': 0,
        'best_val_score': 0.0,
        'training_time': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(1, train_cfg['num_epochs'] + 1):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['num_epochs']}")
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Get edge label index for user-restaurant edges
            edge_label_index = batch[('user', 'rates', 'restaurant')].edge_label_index
            edge_label = batch[('user', 'rates', 'restaurant')].edge_label
            
            # Generate negative samples if using hard sampling
            if hard_sampler is not None:
                edge_label_index, edge_label = hard_sampler.sample_negatives(
                    batch, 
                    edge_label_index, 
                    edge_label,
                    neg_ratio=train_cfg['neg_samples']
                )
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = model(
                batch.x_dict, 
                batch.edge_index_dict, 
                edge_label_index
            )
            
            # Calculate loss
            if loss_type == 'bpr' or loss_type == 'enhanced_bpr':
                # Split positive and negative samples for BPR loss
                pos_mask = edge_label == 1
                pos_scores = out[pos_mask]
                neg_scores = out[~pos_mask]
                loss = criterion(pos_scores, neg_scores)
            elif isinstance(criterion, AdaptiveHybridLoss):
                # For adaptive loss, split positive and negative samples
                pos_mask = edge_label == 1
                pos_scores = out[pos_mask]
                neg_scores = out[~pos_mask]
                loss_dict = criterion(pos_scores, neg_scores)
                loss = loss_dict['loss']  # Extract the total loss
            else:
                # Standard BCE or Focal loss
                loss = criterion(out, edge_label.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip_value:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / max(1, num_batches)
        results['train_loss'].append(avg_loss)
        
        # Log to Weights & Biases
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
        
        # Validation
        if epoch % eval_cfg['validation_frequency'] == 0 or epoch == train_cfg['num_epochs']:
            logger.info(f"Evaluating model at epoch {epoch}...")
            model.eval()
            
            # Compute validation metrics
            val_metrics = validate(model, graph, val_data, k_values=eval_cfg['k_values'], device=device)
            results['val_metrics'].append(val_metrics)
            
            # Extract the metric for the specified k
            val_score = val_metrics[f'ndcg@{k}']
            
            # Log to Weights & Biases
            if use_wandb:
                wandb.log({f"val_{metric}": value for metric, value in val_metrics.items()})
                wandb.log({"epoch": epoch})
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_score)
                else:
                    scheduler.step()
            
            # Check for best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                
                # Save best model if requested
                if train_cfg['save_best_only']:
                    model_path = os.path.join(save_dir, f"{model_type}_{loss_type}_best.pt")
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")
            
            # Print validation metrics
            logger.info(f"Epoch {epoch} | Validation NDCG@{k}: {val_score:.4f} | Best: {best_val_score:.4f} at epoch {best_epoch}")
        
        # Save periodic checkpoints
        if epoch % 10 == 0 and not train_cfg['save_best_only']:
            model_path = os.path.join(save_dir, f"{model_type}_{loss_type}_epoch{epoch}.pt")
            torch.save(model.state_dict(), model_path)
    
    # Calculate total training time
    training_time = time.time() - start_time
    results['training_time'] = training_time
    results['best_epoch'] = best_epoch
    results['best_val_score'] = best_val_score
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation NDCG@{k}: {best_val_score:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    if train_cfg['save_best_only'] and best_epoch < train_cfg['num_epochs']:
        model_path = os.path.join(save_dir, f"{model_type}_{loss_type}_best.pt")
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded best model from {model_path}")
    
    # Save final model if not already saved
    final_model_path = os.path.join(save_dir, f"{model_type}_{loss_type}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Finalize Weights & Biases
    if use_wandb:
        wandb.finish()
    
    return model, (train_data, val_data, test_data), results


def train_all_models(
    graph: HeteroData, 
    config: Optional[Dict[str, Any]] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    k: Optional[int] = None,
    save_dir: str = 'trained_models',
    device: Optional[torch.device] = None,
    use_wandb: bool = False
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Dict[str, float]]]:
    """
    Train all model variants.
    
    Args:
        graph: HeteroData graph
        config: Model configuration
        num_epochs: Number of training epochs (overrides config)
        batch_size: Batch size for training (overrides config)
        k: Value of k for evaluation metrics
        save_dir: Directory to save model checkpoints
        device: Computation device
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Tuple of (dict of trained models, dict of test metrics)
    """
    # Load config if not provided
    if config is None:
        config = cfg.load_config()
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model configurations to train
    model_configs = [
        {'model_type': 'baseline', 'loss_type': 'bce', 'name': 'baseline_bce'},
        {'model_type': 'improved', 'loss_type': 'bce', 'name': 'improved_bce'},
        {'model_type': 'improved', 'loss_type': 'bpr', 'name': 'improved_bpr'},
        {'model_type': 'advanced_gnn', 'loss_type': 'adaptive', 'name': 'advanced_adaptive'},
        # Additional configurations can be added here
    ]
    
    # Train models
    models = {}
    test_metrics = {}
    
    for model_config in model_configs:
        model_type = model_config['model_type']
        loss_type = model_config['loss_type']
        name = model_config['name']
        
        logger.info(f"\n=== Training {name} model ===")
        
        # Initialize new wandb run if enabled
        if use_wandb:
            wandb.init(
                project='yelp-recommendation',
                name=name,
                config=model_config,
                reinit=True
            )
        
        # Train model
        model, (train_data, val_data, test_data), results = train_model(
            graph=graph,
            model_type=model_type,
            loss_type=loss_type,
            config=config,
            num_epochs=num_epochs,
            batch_size=batch_size,
            k=k,
            save_dir=save_dir,
            device=device,
            use_wandb=False  # We're managing wandb runs manually here
        )
        
        # Store model
        models[name] = model
        
        # Evaluate on test data
        logger.info(f"Evaluating {name} model on test data...")
        model.eval()
        metrics = compute_metrics(model, graph, test_data, k_values=[k], device=device)
        test_metrics[name] = metrics
        
        # Log test metrics
        if use_wandb:
            wandb.log({f"test_{metric}": value for metric, value in metrics.items()})
            wandb.finish()
        
        logger.info(f"Test NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    
    # Print comparative results
    logger.info("\n=== Comparative Results ===")
    logger.info(f"{'Model':<20} {'NDCG@' + str(k):<10} {'Recall@' + str(k):<10} {'Precision@' + str(k):<10}")
    logger.info("-" * 50)
    
    for name, metrics in test_metrics.items():
        logger.info(f"{name:<20} {metrics[f'ndcg@{k}']:<10.4f} {metrics[f'recall@{k}']:<10.4f} {metrics[f'precision@{k}']:<10.4f}")
    
    return models, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with processed graph data')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model_type', type=str, default='improved', help='Model type to train')
    parser.add_argument('--loss_type', type=str, default='bce', help='Loss function type')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--k', type=int, default=None, help='Value of k for evaluation metrics')
    parser.add_argument('--hard_sampling', action='store_true', help='Use hard negative sampling')
    parser.add_argument('--save_dir', type=str, default='trained_models', help='Directory to save models')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for tracking')
    parser.add_argument('--train_all', action='store_true', help='Train all model variants')
    
    args = parser.parse_args()
    
    # Load configuration
    config = cfg.load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Load graph data
    graph_path = os.path.join(args.data_dir, 'yelp_graph.pt')
    logger.info(f"Loading graph from {graph_path}...")
    graph = load_tensor(graph_path)
    
    if args.train_all:
        # Train all model variants
        models, test_metrics = train_all_models(
            graph=graph,
            config=config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            k=args.k,
            save_dir=args.save_dir,
            device=device,
            use_wandb=args.wandb
        )
    else:
        # Train a single model
        model, data_splits, results = train_model(
            graph=graph,
            model_type=args.model_type,
            loss_type=args.loss_type,
            config=config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            k=args.k,
            hard_sampling=args.hard_sampling,
            save_dir=args.save_dir,
            device=device,
            use_wandb=args.wandb
        )
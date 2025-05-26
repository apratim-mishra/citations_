"""
Evaluation script for recommendation models.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import argparse

from torch_geometric.data import HeteroData

from models.baseline import BaselineModel
from models.improved import ImprovedModel
from models.advanced_gnn import AdvancedGNNModel
from models.sentence_models import HybridTextGNNModel, TextEmbeddingEncoder

from metrics.ranking import compute_all_ranking_metrics
from utils.preprocessing import split_edge_data, load_tensor, save_tensor
from utils.evaluation import test, compute_metrics, roc_auc_test, precision_recall_curve
from utils.visualization import plot_metrics, plot_precision_recall_curve, plot_model_comparison
from utils.logger import setup_logger
import config.model_config as cfg

# Setup logger
logger = setup_logger()


def load_model(
    model_path: str,
    model_type: str,
    graph: HeteroData,
    config: Dict[str, Any],
    device: torch.device
) -> torch.nn.Module:
    """
    Load a saved model.
    
    Args:
        model_path: Path to the saved model state dict
        model_type: Type of model ('baseline', 'improved', 'advanced_gnn', 'hybrid')
        graph: HeteroData graph
        config: Model configuration
        device: Computation device
        
    Returns:
        Loaded model
    """
    # Get metadata from the graph
    metadata = (graph.node_types, graph.edge_types)
    
    # Get feature dimensions for each node type
    in_channels_dict = {node_type: graph[node_type].x.size(1) for node_type in graph.node_types}
    
    # Initialize the appropriate model
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
        
        # Create text encoder - Note: would need text data to be fully functional
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
            text_weight=0.5,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {model_path}")
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    graph: HeteroData,
    test_data: Tuple,
    k_values: List[int],
    device: torch.device,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Model to evaluate
        graph: HeteroData graph
        test_data: Test data tuple (edge_label_index, edge_label)
        k_values: List of k values for evaluation metrics
        device: Computation device
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Compute standard ranking metrics
    metrics = compute_metrics(model, graph, test_data, k_values=k_values, device=device)
    
    # Compute ROC-AUC
    auc = roc_auc_test(model, graph, test_data, device=device)
    metrics['auc'] = auc
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(model, graph, test_data, device=device)
    
    # Print evaluation results
    logger.info("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save precision-recall curve data
        pr_curve_path = os.path.join(output_dir, 'pr_curve.npz')
        np.savez(pr_curve_path, precision=precision, recall=recall, thresholds=thresholds)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plot_precision_recall_curve(precision, recall, auc)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        
        # Plot metrics comparison
        plt.figure(figsize=(12, 8))
        plot_metrics(metrics, k_values)
        plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved evaluation results to {output_dir}")
    
    return metrics


def evaluate_multiple_models(
    models: Dict[str, torch.nn.Module],
    graph: HeteroData,
    test_data: Tuple,
    k_values: List[int],
    device: torch.device,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models on test data.
    
    Args:
        models: Dictionary of models to evaluate
        graph: HeteroData graph
        test_data: Test data tuple (edge_label_index, edge_label)
        k_values: List of k values for evaluation metrics
        device: Computation device
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary of model names to evaluation metrics
    """
    all_metrics = {}
    
    for model_name, model in models.items():
        logger.info(f"\n=== Evaluating {model_name} model ===")
        
        # Set model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            graph=graph,
            test_data=test_data,
            k_values=k_values,
            device=device,
            output_dir=model_output_dir
        )
        
        all_metrics[model_name] = metrics
    
    # Compare models
    if output_dir and len(models) > 1:
        # Plot model comparison
        plt.figure(figsize=(15, 10))
        plot_model_comparison(all_metrics, k_values)
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Save comparison metrics to CSV
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
        
        logger.info(f"Saved model comparison to {output_dir}")
    
    return all_metrics


def get_top_k_recommendations(
    model: torch.nn.Module,
    graph: HeteroData,
    user_indices: List[int],
    k: int = 10,
    device: torch.device = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Get top-k restaurant recommendations for given users.
    
    Args:
        model: Recommendation model
        graph: HeteroData graph
        user_indices: List of user indices to get recommendations for
        k: Number of recommendations to return
        device: Computation device
        
    Returns:
        Dictionary mapping user indices to lists of (restaurant_idx, score) tuples
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get total number of restaurants
    num_restaurants = graph['restaurant'].x.size(0)
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Store recommendations for each user
    recommendations = {}
    
    # Process each user
    for user_idx in tqdm(user_indices, desc="Generating recommendations"):
        # Create edge_label_index for all possible user-restaurant pairs
        restaurant_indices = torch.arange(num_restaurants, device=device)
        user_indices_tensor = torch.full_like(restaurant_indices, user_idx)
        
        edge_label_index = torch.stack([user_indices_tensor, restaurant_indices])
        
        # Get predictions
        with torch.no_grad():
            scores = model(graph.x_dict, graph.edge_index_dict, edge_label_index)
        
        # Find top-k restaurants
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        # Convert to CPU for further processing
        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        
        # Store results
        recommendations[user_idx] = [(int(idx), float(score)) for idx, score in zip(top_k_indices, top_k_scores)]
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with processed graph data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--model_type', type=str, default='improved', help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, help='Path to specific model file (overrides model_type)')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--k_values', type=str, default='10,50,100', help='Comma-separated list of k values for evaluation')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all models in model_dir')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = cfg.load_config(args.config)
    
    # Load graph data
    graph_path = os.path.join(args.data_dir, 'yelp_graph.pt')
    logger.info(f"Loading graph from {graph_path}...")
    graph = load_tensor(graph_path)
    
    # Prepare test data
    logger.info("Preparing test data...")
    _, _, test_data = split_edge_data(graph, 'user', 'restaurant')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.evaluate_all:
        # Evaluate all models in the model directory
        logger.info(f"Evaluating all models in {args.model_dir}...")
        
        # Find all model files
        model_files = {}
        for file in os.listdir(args.model_dir):
            if file.endswith('.pt'):
                model_name = file.split('.')[0]
                model_type = model_name.split('_')[0]
                model_files[model_name] = {
                    'path': os.path.join(args.model_dir, file),
                    'type': model_type
                }
        
        # Load and evaluate each model
        models = {}
        for model_name, model_info in model_files.items():
            try:
                model = load_model(
                    model_path=model_info['path'],
                    model_type=model_info['type'],
                    graph=graph,
                    config=config,
                    device=device
                )
                models[model_name] = model
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
        
        # Evaluate all loaded models
        all_metrics = evaluate_multiple_models(
            models=models,
            graph=graph,
            test_data=test_data,
            k_values=k_values,
            device=device,
            output_dir=args.output_dir
        )
        
    else:
        # Evaluate a single model
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            # Try to find the final or best model for the specified type
            model_path = os.path.join(args.model_dir, f"{args.model_type}_final.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(args.model_dir, f"{args.model_type}_best.pt")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"No model found for type {args.model_type} in {args.model_dir}")
        
        # Load the model
        model = load_model(
            model_path=model_path,
            model_type=args.model_type,
            graph=graph,
            config=config,
            device=device
        )
        
        # Evaluate the model
        metrics = evaluate_model(
            model=model,
            graph=graph,
            test_data=test_data,
            k_values=k_values,
            device=device,
            output_dir=args.output_dir
        )
        
        # Generate sample recommendations
        logger.info("\nGenerating sample recommendations...")
        num_samples = 5
        sample_users = np.random.choice(graph['user'].x.size(0), num_samples, replace=False).tolist()
        
        recommendations = get_top_k_recommendations(
            model=model,
            graph=graph,
            user_indices=sample_users,
            k=10,
            device=device
        )
        
        # Print sample recommendations
        logger.info(f"\n=== Sample Recommendations for {num_samples} Users ===")
        for user_idx, recs in recommendations.items():
            logger.info(f"User {user_idx}: Top 3 restaurants: {recs[:3]}")
        
        # Save sample recommendations
        recommendations_path = os.path.join(args.output_dir, 'sample_recommendations.json')
        with open(recommendations_path, 'w') as f:
            # Convert to serializable format
            serializable_recs = {str(k): [(int(r[0]), float(r[1])) for r in v] for k, v in recommendations.items()}
            json.dump(serializable_recs, f, indent=4)
        
        logger.info(f"Saved sample recommendations to {recommendations_path}")


if __name__ == "__main__":
    main()
    
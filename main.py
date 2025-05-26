"""
Main entry point for citation graph analysis with link prediction and node regression.
"""
import os
import argparse
import torch
import numpy as np
from typing import Optional, Dict, Any

from data_processor import process_yelp_data
from train import train_model
from train_multitask import train_multitask, train_regression_only
from test import evaluate_model
from utils.preprocessing import load_tensor, save_tensor, split_edge_data, split_node_data
from utils.evaluation import evaluate_regression, evaluate_multitask
from utils.logger import setup_logger
import config.model_config as cfg

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wandb_api_key = os.getenv("WANDB_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")

# Setup logger
logger = setup_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Citation Graph Analysis with Link Prediction and Node Regression')
    
    # General options
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing dataset files')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    
    # Task selection
    parser.add_argument('--task', type=str, 
                       choices=['link_prediction', 'node_regression', 'multitask', 'all'],
                       default='all',
                       help='Task to perform')
    
    # Action selection
    parser.add_argument('--action', type=str, 
                       choices=['process', 'train', 'evaluate', 'all'],
                       default='all',
                       help='Action to perform')
    
    # Processing options
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Force reprocessing of data even if processed files exist')
    parser.add_argument('--rcr_key', type=str, default='relative_citation_ratio',
                       help='Key for RCR values in the paper data')
    
    # Training options
    parser.add_argument('--model_type', type=str, 
                       choices=['baseline', 'improved', 'advanced_gnn', 'hybrid'], 
                       default='improved',
                       help='Type of model to train')
    parser.add_argument('--loss_type', type=str,
                       choices=['bce', 'bpr', 'enhanced_bpr', 'focal', 'adaptive'],
                       default='bce',
                       help='Type of loss function for link prediction')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--mtl_weight', type=float, default=0.5,
                       help='Weight for multi-task learning (0 = only regression, 1 = only link prediction)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for tracking experiments')
    
    # Evaluation options
    parser.add_argument('--k', type=int, default=10,
                       help='Value of k for Recall@k, Precision@k, etc.')
    parser.add_argument('--k_values', type=str, default=None,
                       help='Comma-separated list of k values for evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to a specific model to evaluate')
    
    # Output options
    parser.add_argument('--model_dir', type=str, default='trained_models',
                       help='Directory to save/load trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for training and evaluation."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dir = os.path.join(args.data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load configuration
    config = cfg.load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Parse k values
    k_values = None
    if args.k_values:
        k_values = [int(k) for k in args.k_values.split(',')]
    elif args.k:
        k_values = [args.k]
    
    if k_values:
        config['evaluation']['k_values'] = sorted(list(set(config['evaluation']['k_values'] + k_values)))
    
    return device, config


def process_data(args):
    """Process the dataset."""
    processed_graph_path = os.path.join(args.data_dir, 'processed', 'citation_graph.pt')
    
    if os.path.exists(processed_graph_path) and not args.force_reprocess:
        logger.info(f"Loading processed graph from {processed_graph_path}")
        graph = load_tensor(processed_graph_path)
    else:
        logger.info("Processing citation graph dataset...")
        # This function needs to be implemented for your specific dataset
        # It should load your paper data, author data, citations, and create a HeteroData graph
        graph, _ = process_citation_data(args.data_dir, rcr_key=args.rcr_key)
        
    return graph


def process_citation_data(data_dir, rcr_key='relative_citation_ratio'):
    """
    Process citation data to create a heterogeneous graph.
    This is a placeholder - you'll need to implement this for your specific dataset.
    """
    # Import the appropriate data processor
    from data_processor import process_data_to_graph
    
    # Load your data
    # For example (using dummy data for illustration):
    import pandas as pd
    import numpy as np
    
    # Load author data
    author_df = pd.read_csv(os.path.join(data_dir, 'authors.csv'))
    
    # Load paper data with RCR values
    paper_df = pd.read_csv(os.path.join(data_dir, 'papers.csv'))
    
    # If RCR values don't exist, create dummy values for testing
    if rcr_key not in paper_df.columns:
        logger.warning(f"'{rcr_key}' not found in paper data. Creating dummy values for testing.")
        # Create random RCR values between 0 and 10 with a skewed distribution (most papers have low impact)
        paper_df[rcr_key] = np.random.exponential(scale=1.0, size=len(paper_df))
    
    # Load paper-author relationships
    # Format: {paper_id: [author_ids]}
    paper_author_dict = {}
    # Add your code to populate this dictionary
    
    # Load citation relationships
    # Format: {cited_paper_id: [citing_paper_ids]}
    citation_dict = {}
    # Add your code to populate this dictionary
    
    # Process data to create graph
    graph = process_data_to_graph(
        author_df=author_df,
        paper_df=paper_df,
        paper_author_dict=paper_author_dict,
        citation_dict=citation_dict,
        processed_dir=os.path.join(data_dir, 'processed')
    )
    
    # Save the processed graph
    save_tensor(graph, os.path.join(data_dir, 'processed', 'citation_graph.pt'))
    
    return graph, None


def main():
    args = parse_args()
    
    try:
        # Setup environment
        device, config = setup_environment(args)
        
        # Process data
        if args.action in ['process', 'all']:
            graph = process_data(args)
        else:
            # Load processed graph
            graph_path = os.path.join(args.data_dir, 'processed', 'citation_graph.pt')
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Processed graph not found at {graph_path}. Run with --action process first.")
            
            logger.info(f"Loading graph from {graph_path}...")
            graph = load_tensor(graph_path)
            
            # Print graph info
            logger.info("Graph loaded successfully")
            logger.info(f"Graph node types: {graph.node_types}")
            logger.info(f"Graph edge types: {graph.edge_types}")
            
            # Check if graph has RCR values
            if args.task in ['node_regression', 'multitask', 'all']:
                if args.rcr_key not in graph['paper'].keys:
                    raise ValueError(f"Graph does not have '{args.rcr_key}' attribute for paper nodes")
                logger.info(f"Found {args.rcr_key} values for papers")
        
        # Train models
        if args.action in ['train', 'all']:
            if args.task == 'link_prediction':
                # Train link prediction model
                logger.info(f"Training {args.model_type} model with {args.loss_type} loss for link prediction...")
                model, data_splits, results = train_model(
                    graph=graph,
                    model_type=args.model_type,
                    loss_type=args.loss_type,
                    config=config,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    k=args.k,
                    save_dir=args.model_dir,
                    device=device,
                    use_wandb=args.wandb
                )
                
            elif args.task == 'node_regression':
                # Train node regression model
                logger.info(f"Training {args.model_type} model for {args.rcr_key} prediction...")
                model, results = train_regression_only(
                    graph=graph,
                    rcr_key=args.rcr_key,
                    base_model_type=args.model_type,
                    model_config=config['model'].get(args.model_type),
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    save_dir=args.model_dir,
                    device=device,
                    use_wandb=args.wandb
                )
                
            elif args.task == 'multitask' or args.task == 'all':
                # Train multi-task model
                logger.info(f"Training {args.model_type} model for multi-task learning...")
                model, results = train_multitask(
                    graph=graph,
                    rcr_key=args.rcr_key,
                    base_model_type=args.model_type,
                    model_config=config['model'].get(args.model_type),
                    mtl_weight=args.mtl_weight,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    k=args.k,
                    save_dir=args.model_dir,
                    device=device,
                    use_wandb=args.wandb
                )
        
        # Evaluate models
        if args.action in ['evaluate', 'all']:
            if args.task == 'link_prediction':
                # Evaluate link prediction
                from test import load_model
                
                if args.model_path:
                    model_path = args.model_path
                else:
                    model_path = os.path.join(args.model_dir, f"{args.model_type}_{args.loss_type}_best.pt")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(args.model_dir, f"{args.model_type}_{args.loss_type}_final.pt")
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(f"No model found at expected paths")
                
                logger.info(f"Loading model from {model_path}...")
                model = load_model(
                    model_path=model_path,
                    model_type=args.model_type,
                    graph=graph,
                    config=config,
                    device=device
                )
                
                # Prepare test data for link prediction
                _, _, test_data = split_edge_data(
                    graph, 'author', 'paper', edge_type=('paper', 'cites', 'paper')
                )
                
                # Get k values from config
                k_values = config['evaluation']['k_values']
                
                # Evaluate link prediction
                link_metrics = evaluate_model(
                    model=model,
                    graph=graph,
                    test_data=test_data,
                    k_values=k_values,
                    device=device,
                    output_dir=os.path.join(args.output_dir, 'link_prediction')
                )
                
                logger.info(f"Link prediction results: NDCG@{args.k} = {link_metrics.get(f'ndcg@{args.k}', 0):.4f}")
                
            elif args.task == 'node_regression':
                # Evaluate node regression
                from models.regression import RCRPredictionGNN
                
                model_path = args.model_path or os.path.join(args.model_dir, f"regression_{args.model_type}_best.pt")
                if not os.path.exists(model_path):
                    model_path = os.path.join(args.model_dir, f"regression_{args.model_type}_final.pt")
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"No regression model found at expected paths")
                
                logger.info(f"Loading regression model from {model_path}...")
                # This requires implementing a load_regression_model function
                model = load_regression_model(
                    model_path=model_path,
                    base_model_type=args.model_type,
                    graph=graph,
                    config=config,
                    device=device
                )
                
                # Prepare test data for node regression
                _, _, test_mask = split_node_data(
                    graph, 'paper', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
                )
                
                test_indices = torch.arange(graph['paper'].num_nodes)[test_mask]
                test_values = graph['paper'][args.rcr_key][test_mask].float()
                
                # Evaluate regression
                regression_metrics = evaluate_regression(
                    model=model,
                    graph=graph,
                    node_indices=test_indices,
                    true_values=test_values,
                    device=device
                )
                
                # Save results
                import json
                os.makedirs(os.path.join(args.output_dir, 'node_regression'), exist_ok=True)
                with open(os.path.join(args.output_dir, 'node_regression', 'metrics.json'), 'w') as f:
                    json.dump(regression_metrics, f, indent=2)
                
                logger.info(f"Node regression results: MSE = {regression_metrics['mse']:.4f}, R² = {regression_metrics['r2']:.4f}")
                
            elif args.task == 'multitask' or args.task == 'all':
                # Evaluate multi-task model
                from models.regression import MultiTaskGNN
                
                model_path = args.model_path or os.path.join(args.model_dir, f"multitask_{args.model_type}_best.pt")
                if not os.path.exists(model_path):
                    model_path = os.path.join(args.model_dir, f"multitask_{args.model_type}_final.pt")
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"No multi-task model found at expected paths")
                
                logger.info(f"Loading multi-task model from {model_path}...")
                # This requires implementing a load_multitask_model function
                model = load_multitask_model(
                    model_path=model_path,
                    base_model_type=args.model_type,
                    graph=graph,
                    config=config,
                    device=device
                )
                
                # Prepare test data for link prediction
                _, _, link_test_data = split_edge_data(
                    graph, 'author', 'paper', edge_type=('paper', 'cites', 'paper')
                )
                
                # Prepare test data for node regression
                _, _, test_mask = split_node_data(
                    graph, 'paper', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
                )
                
                test_indices = torch.arange(graph['paper'].num_nodes)[test_mask]
                test_values = graph['paper'][args.rcr_key][test_mask].float()
                
                # Get k values from config
                k_values = config['evaluation']['k_values']
                
                # Evaluate multi-task model
                multitask_metrics = evaluate_multitask(
                    model=model,
                    graph=graph,
                    link_test_data=link_test_data,
                    regression_test_indices=test_indices,
                    regression_test_values=test_values,
                    k_values=k_values,
                    device=device
                )
                
                # Save results
                import json
                os.makedirs(os.path.join(args.output_dir, 'multitask'), exist_ok=True)
                with open(os.path.join(args.output_dir, 'multitask', 'metrics.json'), 'w') as f:
                    # Convert any non-serializable values to strings
                    serializable_metrics = {}
                    for key, value in multitask_metrics.items():
                        if isinstance(value, dict):
                            serializable_metrics[key] = {k: float(v) for k, v in value.items()}
                        else:
                            serializable_metrics[key] = float(value)
                    json.dump(serializable_metrics, f, indent=2)
                
                # Print summary results
                link_ndcg = multitask_metrics['link_prediction'].get(f'ndcg@{args.k}', 0)
                reg_mse = multitask_metrics['regression']['mse']
                reg_r2 = multitask_metrics['regression']['r2']
                
                logger.info(f"Multi-task results:")
                logger.info(f"  Link prediction: NDCG@{args.k} = {link_ndcg:.4f}")
                logger.info(f"  Node regression: MSE = {reg_mse:.4f}, R² = {reg_r2:.4f}")
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def load_regression_model(model_path, base_model_type, graph, config, device):
    """
    Load a regression model.
    This is a placeholder - you'll need to implement this for your specific models.
    """
    from test import load_model
    from models.regression import RCRPredictionGNN
    
    # Load base GNN model
    base_model = load_model(
        model_path=os.path.join(config['model_dir'], f"{base_model_type}_final.pt"),
        model_type=base_model_type,
        graph=graph,
        config=config,
        device=device
    )
    
    # Create regression model
    model = RCRPredictionGNN(
        gnn_model=base_model,
        embedding_dim=config['model'][base_model_type].get('out_channels', 64),
        hidden_dim=config['model'][base_model_type].get('hidden_channels', 128),
        num_layers=config['model'][base_model_type].get('num_layers', 3),
        dropout=config['model'][base_model_type].get('dropout', 0.3),
        target_node_type='paper',
        device=device
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def load_multitask_model(model_path, base_model_type, graph, config, device):
    """
    Load a multi-task model.
    This is a placeholder - you'll need to implement this for your specific models.
    """
    from test import load_model
    from models.regression import MultiTaskGNN
    
    # Load base GNN model
    base_model = load_model(
        model_path=os.path.join(config['model_dir'], f"{base_model_type}_final.pt"),
        model_type=base_model_type,
        graph=graph,
        config=config,
        device=device
    )
    
    # Create multi-task model
    model = MultiTaskGNN(
        gnn_model=base_model,
        node_type='paper',
        regression_hidden_channels=config['model'][base_model_type].get('hidden_channels', 64),
        regression_num_layers=config['model'][base_model_type].get('num_layers', 2),
        regression_dropout=config['model'][base_model_type].get('dropout', 0.2),
        mtl_weight=0.5,  # Default value, not important for evaluation
        device=device
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
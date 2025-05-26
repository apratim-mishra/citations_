"""
Configuration parameters for recommendation models.
"""
import os
import yaml
from typing import Dict, Any, Optional, List, Union

# Default configuration
DEFAULT_CONFIG = {
    # Model parameters
    "model": {
        "baseline": {
            "hidden_channels": 64,
            "out_channels": 64,
            "num_layers": 2,
            "dropout": 0.3
        },
        "improved": {
            "hidden_channels": 64,
            "out_channels": 64,
            "num_layers": 3,
            "dropout": 0.2
        },
        "sentence": {
            "model_name": "all-MiniLM-L6-v2",
            "pooling_mode": "mean",
            "max_seq_length": 128
        },
        "advanced_gnn": {
            "hidden_channels": 128,
            "out_channels": 64,
            "num_layers": 4,
            "dropout": 0.2,
            "residual": True,
            "layer_norm": True,
            "attention_heads": 4
        },
        "hybrid": {
            "text_hidden_dim": 256,
            "gnn_hidden_dim": 128,
            "combined_dim": 64,
            "dropout": 0.2
        },
        "llm": {
            "model_type": "huggingface",  # or "openai"
            "huggingface_model": "sentence-transformers/all-mpnet-base-v2",
            "openai_model": "text-embedding-ada-002",
            "openai_api_key": None,  # Should be set through environment variable
            "embedding_dimension": 768
        }
    },

    # Training parameters
    "training": {
        "num_epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10,
        "scheduler": "reduce_on_plateau",  # 'cosine', 'step', 'reduce_on_plateau'
        "scheduler_params": {
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        },
        "gradient_clipping": 1.0,
        "save_best_only": True,
        "neg_samples": 10,  # Number of negative samples per positive sample
        "hard_sampling": False  # Whether to use hard negative sampling
    },

    # Loss function parameters
    "loss": {
        "type": "bce",  # 'bce', 'bpr', 'contrastive', 'adaptive'
        "contrastive": {
            "margin": 0.5,
            "temperature": 0.1
        },
        "adaptive": {
            "alpha": 0.25,
            "gamma": 2.0,
            "beta": 0.9
        }
    },
    
    # Evaluation parameters
    "evaluation": {
        "k_values": [10, 50, 100, 300],  # k values for Precision@k, Recall@k
        "validation_frequency": 5,  # Validate every N epochs
        "metrics": ["map", "ndcg", "recall", "precision", "auc"]
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and override defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Recursively update the default config
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
                
            config = update_dict(config, user_config)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
    
    # Override with environment variables if set
    if os.environ.get("OPENAI_API_KEY"):
        config["model"]["llm"]["openai_api_key"] = os.environ["OPENAI_API_KEY"]
        
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")
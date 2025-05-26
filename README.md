# Citation Graph Analysis Framework

A comprehensive framework for analyzing academic citation networks using Graph Neural Networks. This framework supports multi-task learning to simultaneously predict citation links and paper impact metrics.

## Features

- **Link Prediction**: Predict citation relationships between papers
- **Node Regression**: Predict Relative Citation Ratio (RCR) for papers
- **Multi-Task Learning**: Joint training of both link prediction and impact prediction
- **Multiple GNN Architectures**: Baseline, Improved, and Advanced GNN models
- **Advanced Loss Functions**: BCE, BPR, Focal Loss, and Adaptive Loss options
- **Hard Negative Sampling**: Improved training with challenging negative examples
- **External API Integration**: Support for HuggingFace and OpenAI models
- **Comprehensive Evaluation Metrics**: NDCG, MAP, Precision/Recall for link prediction; MSE, MAE, R² for regression

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Tasks](#tasks)
- [Models](#models)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/citation-graph-analysis.git
cd citation-graph-analysis

# Install dependencies
pip install -r requirements.txt

# then install pyg

pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html




# Verify your environment
python test_python.py


# setup data

python setup_data.py --author-file /path/to/merged_auids.csv \
                     --paper-file /path/to/merged_df_cleaned.csv \
                     --citation-file /path/to/citing_articles_cleaned.json
```

### Requirements

Core dependencies:
- Python 3.7+
- PyTorch 2.0+
- PyTorch Geometric 2.6+
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib
- wandb (optional, for experiment tracking)

## Project Structure

```
citation-graph/
├── main.py                    # Main entry point
├── data_processor.py          # Data processing pipeline
├── train.py                   # Link prediction training
├── train_multitask.py         # Multi-task training
├── test.py                    # Evaluation functions
├── api/                       # External API integrations
│   ├── huggingface_client.py  # HuggingFace API
│   └── openai_client.py       # OpenAI API
├── config/                    # Configuration files
│   └── model_config.py        # Model and training parameters
├── losses/                    # Loss functions
│   ├── bpr_loss.py            # Bayesian Personalized Ranking
│   ├── adaptive_loss.py       # Adaptive loss functions
│   └── ...                    # Other loss implementations
├── models/                    # Model architectures
│   ├── baseline.py            # Simple GNN models
│   ├── improved.py            # Enhanced GNN models
│   ├── advanced_gnn.py        # Advanced GNN architectures
│   ├── sentence_models.py     # Text-based models
│   ├── regression.py          # RCR prediction models
│   └── ...                    # Other model implementations
├── utils/                     # Utility functions
│   ├── preprocessing.py       # Data preprocessing
│   ├── evaluation.py          # Evaluation metrics
│   ├── visualization.py       # Result visualization
│   ├── sampling.py            # Negative sampling strategies
│   └── ...                    # Other utilities
```

## Quick Start

### Process Data

```bash

python data_processor.py --data-dir ./data --output-dir ./processed_graph_data


python main.py --action process --data_dir path/to/your/data
```

### Train a Model

```bash
# Link prediction only
python main.py --task link_prediction --model_type improved --loss_type bpr --cuda

# Node regression only (predicting RCR)
python main.py --task node_regression --model_type advanced_gnn --cuda

# Multi-task (both link prediction and regression)
python main.py --task multitask --model_type improved --mtl_weight 0.5 --cuda
```

### Evaluate a Model

```bash
python main.py --action evaluate --task multitask --model_path trained_models/multitask_improved_best.pt
```

## Tasks

### Link Prediction

Predicts citation relationships between papers. The model learns to identify which papers are likely to cite others based on their content, authors, and existing citation patterns.

```bash
python main.py --task link_prediction --model_type improved --loss_type bpr --epochs 100
```

### Node Regression (RCR Prediction)

Predicts the Relative Citation Ratio (RCR) of papers, which is a normalized measure of citation impact. This helps identify high-impact papers based on their attributes and network position.

```bash
python main.py --task node_regression --model_type advanced_gnn --epochs 100
```

### Multi-Task Learning

Simultaneously predicts both citation links and RCR values, allowing the model to leverage information from both tasks.

```bash
python main.py --task multitask --model_type improved --mtl_weight 0.5 --epochs 100
```

The `mtl_weight` parameter controls the balance between tasks:
- `mtl_weight=1.0`: Focus entirely on link prediction
- `mtl_weight=0.0`: Focus entirely on regression
- `mtl_weight=0.5`: Equal weight to both tasks

## Models

### Baseline Models

Simple GNN models with basic message passing and limited feature processing.

### Improved Models

Enhanced GNNs with better edge decoding, batch normalization, and skip connections.

```python
class ImprovedModel(torch.nn.Module):
    """
    Improved recommendation model with enhanced GNN architecture.
    """
    def __init__(self, metadata, hidden_channels=64, device=None):
        # ...implementation details
```

### Advanced GNN Models

Sophisticated GNN architectures with:
- Multiple message passing layers
- Skip connections
- Layer normalization
- Attention mechanisms
- Residual connections

```bash
python main.py --task multitask --model_type advanced_gnn --cuda
```

### Regression Models

Models specifically designed for predicting continuous RCR values:

- `NodeRegressor`: Simple MLP for regression
- `RCRPredictionGNN`: GNN model optimized for RCR prediction
- `MultiTaskGNN`: Joint model for both link prediction and RCR prediction

## Configuration

The project uses a flexible configuration system in `config/model_config.py`. You can override these settings via command-line arguments or by creating a custom YAML configuration file.

```bash
python main.py --config my_custom_config.yaml
```

### Example Configuration

```yaml
model:
  advanced_gnn:
    hidden_channels: 128
    out_channels: 64
    num_layers: 4
    dropout: 0.2
    residual: true
    layer_norm: true
    attention_heads: 4

training:
  num_epochs: 100
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 1e-5
  early_stopping_patience: 10

loss:
  type: "adaptive"
  adaptive:
    alpha: 0.25
    gamma: 2.0
    beta: 0.5
```

## Evaluation

The framework provides comprehensive evaluation for both link prediction and regression tasks.

### Link Prediction Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Precision@k** and **Recall@k**: Precision and recall at rank k
- **MRR**: Mean Reciprocal Rank
- **AUC**: Area Under the ROC Curve

### Regression Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Explained Variance**

```bash
# Get detailed evaluation with multiple k values
python main.py --action evaluate --task multitask --k_values 10,20,50,100
```

## Advanced Usage

### Hard Negative Sampling

Improve training by using a reference model to generate challenging negative examples:

```bash
python main.py --task link_prediction --model_type improved --hard_sampling --cuda
```

### Using External APIs

For enhanced text-based recommendations or explanations:

```bash
# Using HuggingFace models
python main.py --task link_prediction --use_api --api_type huggingface

# Using OpenAI models (requires API key)
export OPENAI_API_KEY=your_api_key
python main.py --task link_prediction --use_api --api_type openai
```

### Experiment Tracking

Track experiments with Weights & Biases:

```bash
python main.py --task multitask --model_type advanced_gnn --wandb
```

### Custom Loss Functions

Use different loss functions for better performance:

```bash
# BPR loss for better ranking
python main.py --task link_prediction --loss_type bpr

# Enhanced BPR with regularization
python main.py --task link_prediction --loss_type enhanced_bpr

# Adaptive loss for imbalanced data
python main.py --task link_prediction --loss_type adaptive
```


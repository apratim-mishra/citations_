"""
Metrics for evaluating link prediction and node regression.
"""
from .ranking import precision_at_k, recall_at_k, mean_average_precision, ndcg_at_k
from .ranking import mean_reciprocal_rank, compute_all_ranking_metrics
from .classification import auc_score, precision_recall_auc, f1_score
from .classification import accuracy, classification_report
from .distance import cosine_similarity, euclidean_distance, manhattan_distance

__all__ = [
    # Ranking metrics
    'precision_at_k',
    'recall_at_k',
    'mean_average_precision',
    'ndcg_at_k',
    'mean_reciprocal_rank',
    'compute_all_ranking_metrics',
    
    # Classification metrics
    'auc_score',
    'precision_recall_auc',
    'f1_score',
    'accuracy',
    'classification_report',
    
    # Distance metrics
    'cosine_similarity',
    'euclidean_distance',
    'manhattan_distance'
]
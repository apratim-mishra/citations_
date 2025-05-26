"""
Data management utilities for consistent data loading and paths.
"""
import os
import json
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np

class DataPaths:
    """Class to manage data paths consistently throughout the project."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize data paths.
        
        Args:
            base_dir: Base directory (defaults to DATA_DIR from environment or './data')
        """
        # Set base directory from argument, environment variable, or default
        self.base_dir = base_dir or os.environ.get('DATA_DIR', './data')
        
        # Create main data directories
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.interim_dir = os.path.join(self.base_dir, 'interim')
        self.processed_dir = os.path.join(self.base_dir, 'processed')
        self.embeddings_dir = os.path.join(self.base_dir, 'embeddings')
        self.external_dir = os.path.join(self.base_dir, 'external')
        
        # Create subdirectories
        self._create_directories()
        
        # Create path dictionary for easy access
        self.paths = {
            'raw': self.raw_dir,
            'interim': self.interim_dir,
            'processed': self.processed_dir,
            'embeddings': self.embeddings_dir,
            'external': self.external_dir,
            'text_embeddings': os.path.join(self.embeddings_dir, 'text_embeddings'),
        }
    
    def _create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        for path in self.paths.values() if hasattr(self, 'paths') else [
            self.raw_dir, self.interim_dir, self.processed_dir, 
            self.embeddings_dir, self.external_dir,
            os.path.join(self.embeddings_dir, 'text_embeddings')
        ]:
            os.makedirs(path, exist_ok=True)
    
    def get_path(self, data_type: str, filename: Optional[str] = None) -> str:
        """
        Get the path for a specific data type and filename.
        
        Args:
            data_type: Type of data ('raw', 'interim', 'processed', etc.)
            filename: Optional filename to append to the path
            
        Returns:
            Full path
        """
        if data_type not in self.paths:
            raise ValueError(f"Invalid data type: {data_type}. Valid options: {list(self.paths.keys())}")
        
        if filename:
            return os.path.join(self.paths[data_type], filename)
        return self.paths[data_type]


class CitationDataLoader:
    """Utility class for loading citation graph data."""
    
    def __init__(self, data_paths: Optional[DataPaths] = None):
        """
        Initialize data loader.
        
        Args:
            data_paths: DataPaths object (creates a new one if None)
        """
        self.data_paths = data_paths or DataPaths()
    
    def load_author_data(self, filename: str = 'merged_auids.csv') -> pd.DataFrame:
        """
        Load author data.
        
        Args:
            filename: Name of author data file
            
        Returns:
            Author DataFrame
        """
        author_path = self.data_paths.get_path('raw', filename)
        
        if not os.path.exists(author_path):
            raise FileNotFoundError(f"Author data file not found at {author_path}")
        
        return pd.read_csv(author_path)
    
    def load_paper_data(self, filename: str = 'merged_df_cleaned.csv') -> pd.DataFrame:
        """
        Load paper data.
        
        Args:
            filename: Name of paper data file
            
        Returns:
            Paper DataFrame
        """
        paper_path = self.data_paths.get_path('raw', filename)
        
        if not os.path.exists(paper_path):
            raise FileNotFoundError(f"Paper data file not found at {paper_path}")
        
        return pd.read_csv(paper_path)
    
    def load_citation_data(self, filename: str = 'citing_articles_cleaned.json') -> Dict:
        """
        Load citation data.
        
        Args:
            filename: Name of citation data file
            
        Returns:
            Citation dictionary
        """
        citation_path = self.data_paths.get_path('raw', filename)
        
        if not os.path.exists(citation_path):
            raise FileNotFoundError(f"Citation data file not found at {citation_path}")
        
        with open(citation_path, 'r') as f:
            citation_data = json.load(f)
        
        return citation_data
    
    def load_processed_graph(self, filename: str = 'citation_graph.pt') -> Any:
        """
        Load processed PyTorch Geometric graph.
        
        Args:
            filename: Name of graph file
            
        Returns:
            PyTorch Geometric graph object
        """
        graph_path = self.data_paths.get_path('processed', filename)
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Processed graph not found at {graph_path}")
        
        return torch.load(graph_path)
    
    def build_paper_author_dict(
        self, 
        author_df: pd.DataFrame, 
        paper_df: pd.DataFrame,
        paper_id_col: str = 'pmid',
        author_id_col: str = 'AUID',
        paper_author_col: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Build paper to author mapping.
        This is a placeholder - you'll need to adapt this to your specific data structure.
        
        Args:
            author_df: Author DataFrame
            paper_df: Paper DataFrame
            paper_id_col: Column name for paper IDs
            author_id_col: Column name for author IDs
            paper_author_col: Optional column name linking papers to authors
            
        Returns:
            Dictionary mapping paper IDs to lists of author IDs
        """
        # This is a placeholder - modify based on your actual data structure
        # You might have a separate table linking papers to authors
        
        if paper_author_col:
            # If you have a column in the author_df that specifies which paper they authored
            paper_author_dict = {}
            for paper_id in paper_df[paper_id_col].unique():
                paper_id_str = str(paper_id)
                author_ids = author_df[author_df[paper_author_col] == paper_id][author_id_col].tolist()
                paper_author_dict[paper_id_str] = [str(author_id) for author_id in author_ids]
            
            return paper_author_dict
        else:
            # Create a dummy/random mapping for demonstration
            # In a real scenario, replace this with your actual paper-author mapping logic
            print("WARNING: Creating dummy paper-author mappings. Replace with actual mapping logic.")
            
            # Simplified implementation - randomly assign authors to papers
            paper_author_dict = {}
            paper_ids = paper_df[paper_id_col].unique()
            author_ids = author_df[author_id_col].unique()
            
            for paper_id in paper_ids:
                paper_id_str = str(paper_id)
                # Assign 1-5 random authors to each paper
                num_authors = np.random.randint(1, min(6, len(author_ids)))
                selected_authors = np.random.choice(author_ids, size=num_authors, replace=False)
                paper_author_dict[paper_id_str] = [str(author_id) for author_id in selected_authors]
            
            return paper_author_dict
    
    def process_citation_dict(
        self, 
        citation_data: Dict, 
        paper_df: pd.DataFrame,
        paper_id_col: str = 'pmid'
    ) -> Dict[str, List[str]]:
        """
        Process citation data into a dictionary format for graph construction.
        
        Args:
            citation_data: Raw citation data
            paper_df: Paper DataFrame for filtering citations
            paper_id_col: Column name for paper IDs
            
        Returns:
            Dictionary mapping cited paper IDs to lists of citing paper IDs
        """
        # This is a placeholder - modify based on your actual data structure
        
        # Convert paper IDs to strings for consistent handling
        paper_ids = set(str(pid) for pid in paper_df[paper_id_col])
        
        # Process citation data into the required format:
        # {cited_paper_id: [citing_paper_id1, citing_paper_id2, ...]}
        filtered_dict = {}
        
        for cited_id, citing_ids in citation_data.items():
            cited_id_str = str(cited_id)
            
            # Skip if the cited paper is not in our paper dataset
            if cited_id_str not in paper_ids:
                continue
            
            # Filter citing papers to only include those in our paper dataset
            valid_citing_ids = [str(cid) for cid in citing_ids if str(cid) in paper_ids]
            
            if valid_citing_ids:
                filtered_dict[cited_id_str] = valid_citing_ids
        
        return filtered_dict
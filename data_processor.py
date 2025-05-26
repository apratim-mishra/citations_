"""
Data processing pipeline for citation network data.
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import warnings
import json

from utils.data_management import DataPaths, CitationDataLoader
from utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Suppress SettingWithCopyWarning for cleaner output during processing
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Configuration ---
TEXT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_categorical_features(df, columns, nan_placeholder='_NaN_'):
    """Encodes specified categorical columns using LabelEncoder. Handles NaNs."""
    encoders = {}
    df_encoded = df.copy()
    for col in tqdm(columns, desc="Encoding categorical features"):
        # Fill NaNs with a placeholder string before encoding
        df_encoded[col] = df_encoded[col].fillna(nan_placeholder).astype(str)
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col])
        encoders[col] = encoder # Store encoder if needed later
    return df_encoded, encoders


def generate_text_embeddings(texts, model_name=TEXT_EMBEDDING_MODEL, device=DEVICE, batch_size=32):
    """Generates embeddings for a list of texts using SentenceTransformer."""
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print("Generating text embeddings...")
    # Handle potential NaN/None values in text - replace with empty string
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=device)
    # Return tensor on CPU to match other tensors
    return torch.tensor(embeddings, dtype=torch.float).cpu()


def save_data(data, filepath):
    """Saves data (tensor or mapping) to a file."""
    print(f"Saving data to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(data, torch.Tensor):
        torch.save(data, filepath)
    elif isinstance(data, dict) or isinstance(data, pd.DataFrame) or isinstance(data, list):
        torch.save(data, filepath)
    else:
        print(f"Warning: Unsupported data type for saving: {type(data)}")


def load_data(filepath):
    """Loads data (tensor or mapping) from a file."""
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        return torch.load(filepath)
    else:
        print(f"File not found: {filepath}")
        return None


def preprocess_authors(unique_auids_df, save_dir=None):
    """Processes author data, encodes features, creates mapping."""
    print("Processing authors...")
    processed_path = os.path.join(save_dir, 'author_features.pt') if save_dir else None
    mapping_path = os.path.join(save_dir, 'author_mapping.pt') if save_dir else None

    if save_dir and os.path.exists(processed_path) and os.path.exists(mapping_path):
        print("Loading preprocessed author data...")
        author_features = load_data(processed_path)
        auid_to_idx = load_data(mapping_path)
        if author_features is not None and auid_to_idx is not None:
             return author_features, auid_to_idx
        else:
            print("Failed to load preprocessed author data, reprocessing...")

    # Ensure AUID is string
    unique_auids_df['AUID'] = unique_auids_df['AUID'].astype(str)

    # Create mapping from AUID to a continuous index
    auid_list = unique_auids_df['AUID'].tolist()
    auid_to_idx = {auid: i for i, auid in enumerate(auid_list)}

    # Select and encode categorical features
    author_cat_cols = ['Genni', 'gender_guessor', 'Ethnea', 'ethniccolr']
    # Ensure columns exist
    author_cat_cols = [col for col in author_cat_cols if col in unique_auids_df.columns]
    if not author_cat_cols:
         print("Warning: No categorical author feature columns found. Author features will be empty.")
         author_features = torch.empty((len(auid_to_idx), 0), dtype=torch.float) # Empty features
    else:
        authors_encoded_df, _ = encode_categorical_features(unique_auids_df, author_cat_cols)
        # Extract features corresponding to the mapping order
        # Set index to AUID to easily align with auid_list
        authors_encoded_df = authors_encoded_df.set_index('AUID')
        # Reindex based on the auid_list to ensure correct order
        authors_encoded_df = authors_encoded_df.reindex(auid_list)
        author_features = torch.tensor(authors_encoded_df[author_cat_cols].values, dtype=torch.float)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_data(author_features, processed_path)
        save_data(auid_to_idx, mapping_path)

    return author_features, auid_to_idx


def preprocess_papers(merged_df, pmid_col='pmid', rcr_key='relative_citation_ratio', save_dir=None):
    """Processes paper data, encodes/embeds features, creates mapping."""
    print("Processing papers...")
    features_path = os.path.join(save_dir, 'paper_features.pt') if save_dir else None
    mapping_path = os.path.join(save_dir, 'paper_mapping.pt') if save_dir else None
    rcr_path = os.path.join(save_dir, 'paper_rcr.pt') if save_dir else None

    if save_dir and os.path.exists(features_path) and os.path.exists(mapping_path):
        print("Loading preprocessed paper data...")
        paper_features_combined = load_data(features_path)
        pmid_to_idx = load_data(mapping_path)
        if paper_features_combined is not None and pmid_to_idx is not None:
            # Also check for RCR values
            if os.path.exists(rcr_path):
                paper_rcr = load_data(rcr_path)
                return paper_features_combined, pmid_to_idx, paper_rcr
            return paper_features_combined, pmid_to_idx, None
        else:
            print("Failed to load preprocessed paper data, reprocessing...")

    # Ensure PMID column exists and is string
    if pmid_col not in merged_df.columns:
        raise ValueError(f"PMID column '{pmid_col}' not found in paper DataFrame.")
    merged_df[pmid_col] = merged_df[pmid_col].astype(str)

    # Create mapping from PMID to a continuous index
    pmid_list = merged_df[pmid_col].unique().tolist()
    pmid_to_idx = {pmid: i for i, pmid in enumerate(pmid_list)}
    
    # Keep only one row per pmid for feature extraction (if duplicates exist)
    paper_df_unique = merged_df.drop_duplicates(subset=[pmid_col]).set_index(pmid_col)
    # Reindex to match the pmid_to_idx order
    paper_df_unique = paper_df_unique.reindex(pmid_list)

    # Extract RCR values if available
    paper_rcr = None
    if rcr_key in paper_df_unique.columns:
        paper_rcr = torch.tensor(paper_df_unique[rcr_key].fillna(0).values, dtype=torch.float)
        if save_dir:
            save_data(paper_rcr, rcr_path)

    # --- Feature Extraction ---
    all_feature_tensors = []

    # 1. Numerical Features
    num_cols = ['year', 'no_authors', 'International', 'prior_cit_mean',
                'grids_max_yr', 'English_speaking', 'hype_percentile', 'hype_value',
                'ethnic_diversity', 'gender_diversity', 'age_diversity',
                'expertise_diversity', 'abstract_length']
    # Filter out columns not present in the dataframe
    num_cols = [col for col in num_cols if col in paper_df_unique.columns]
    if num_cols:
        print(f"Processing numerical features: {num_cols}")
        numerical_features = paper_df_unique[num_cols].fillna(0).astype(np.float32) # Fill NaNs with 0
        all_feature_tensors.append(torch.tensor(numerical_features.values, dtype=torch.float))
    else:
        print("Warning: No numerical paper feature columns found.")

    # 2. Categorical Features (Label Encoded)
    cat_cols = ['journal', 'journal_type', 'Country', 'hype_word']
     # Filter out columns not present
    cat_cols = [col for col in cat_cols if col in paper_df_unique.columns]
    if cat_cols:
        print(f"Processing categorical features: {cat_cols}")
        papers_encoded_df, _ = encode_categorical_features(paper_df_unique.reset_index(), cat_cols)
        # Need to align back with pmid_to_idx order
        papers_encoded_df = papers_encoded_df.set_index(pmid_col).reindex(pmid_list)
        categorical_features = torch.tensor(papers_encoded_df[cat_cols].values, dtype=torch.float)
        all_feature_tensors.append(categorical_features)
    else:
        print("Warning: No categorical paper feature columns found for label encoding.")

    # 3. Text Features (Embeddings)
    text_cols = ['title', 'abstract', 'sentence']
    # Filter out columns not present
    text_cols = [col for col in text_cols if col in paper_df_unique.columns]
    if text_cols:
        print(f"Processing text features: {text_cols}")
        for col in text_cols:
            texts = paper_df_unique[col].tolist()
            embeddings = generate_text_embeddings(texts)
            all_feature_tensors.append(embeddings)
    else:
        print("Warning: No text paper feature columns found for embedding.")

    # --- Combine Features ---
    if not all_feature_tensors:
         print("Warning: No paper features were generated. Paper features will be empty.")
         # Create empty features tensor matching the number of papers
         paper_features_combined = torch.empty((len(pmid_to_idx), 0), dtype=torch.float, device=DEVICE)
    else:
        # Concatenate all feature tensors horizontally
        paper_features_combined = torch.cat(all_feature_tensors, dim=1)

    # Move final tensor to CPU for saving if it was on GPU
    paper_features_combined = paper_features_combined.cpu()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_data(paper_features_combined, features_path)
        save_data(pmid_to_idx, mapping_path)

    return paper_features_combined, pmid_to_idx, paper_rcr


def create_heterogeneous_graph(
    author_features, 
    paper_features,
    auid_to_idx, 
    pmid_to_idx,
    pmid_auid_dict, 
    filtered_dict,
    paper_rcr=None,
    save_dir=None
):
    """Creates the HeteroData graph object."""
    print("Creating heterogeneous graph...")
    graph_path = os.path.join(save_dir, 'hetero_graph.pt') if save_dir else None

    if save_dir and os.path.exists(graph_path):
        print("Loading preprocessed graph...")
        graph = load_data(graph_path)
        if graph is not None:
            return graph
        else:
            print("Failed to load preprocessed graph, reprocessing...")

    graph = HeteroData()

    # Add node features
    # Ensure features are float tensors
    graph['author'].x = author_features.float()
    graph['paper'].x = paper_features.float()
    
    # Add RCR values as a node property if available
    if paper_rcr is not None:
        graph['paper'].y = paper_rcr.float()
    
    print(f"Author nodes: {graph['author'].num_nodes}, Features: {graph['author'].num_features}")
    print(f"Paper nodes: {graph['paper'].num_nodes}, Features: {graph['paper'].num_features}")

    # --- Add Edges ---

    # 1. Author -> Writes -> Paper edges
    print("Processing 'writes' edges...")
    writes_source = []
    writes_target = []
    valid_writes_edges = 0
    for pmid, auids in tqdm(pmid_auid_dict.items(), desc="Author-Paper Edges"):
        pmid_str = str(pmid) # Ensure pmid is string
        if pmid_str in pmid_to_idx:
            paper_idx = pmid_to_idx[pmid_str]
            for auid in auids:
                auid_str = str(auid) # Ensure auid is string
                if auid_str in auid_to_idx:
                    author_idx = auid_to_idx[auid_str]
                    writes_source.append(author_idx)
                    writes_target.append(paper_idx)
                    valid_writes_edges += 1

    if writes_source:
        graph['author', 'writes', 'paper'].edge_index = torch.tensor([writes_source, writes_target], dtype=torch.long)
        print(f"Added {valid_writes_edges} 'writes' edges.")
    else:
        print("Warning: No valid 'writes' edges were created.")

    # 2. Paper -> Cites -> Paper edges
    # filtered_dict: key = cited paper (target), value = list of citing papers (source)
    print("Processing 'cites' edges...")
    cites_source = []
    cites_target = []
    valid_cites_edges = 0
    for cited_pmid, citing_pmids in tqdm(filtered_dict.items(), desc="Paper-Paper Edges"):
         cited_pmid_str = str(cited_pmid)
         if cited_pmid_str in pmid_to_idx:
             cited_paper_idx = pmid_to_idx[cited_pmid_str]
             for citing_pmid in citing_pmids:
                 citing_pmid_str = str(citing_pmid)
                 if citing_pmid_str in pmid_to_idx:
                     citing_paper_idx = pmid_to_idx[citing_pmid_str]
                     # Edge: citing_paper -> cites -> cited_paper
                     cites_source.append(citing_paper_idx)
                     cites_target.append(cited_paper_idx)
                     valid_cites_edges += 1

    if cites_source:
        graph['paper', 'cites', 'paper'].edge_index = torch.tensor([cites_source, cites_target], dtype=torch.long)
        print(f"Added {valid_cites_edges} 'cites' edges.")
    else:
        print("Warning: No valid 'cites' edges were created.")

    if save_dir:
        save_data(graph, graph_path)

    return graph


def process_data_to_graph(
    data_dir: str,
    author_file: str = 'merged_auids.csv',
    paper_file: str = 'merged_df_cleaned.csv',
    citation_file: str = 'citing_articles_cleaned.json',
    processed_dir: str = 'processed_graph_data',
    force_reprocess: bool = False
):
    """
    Main function to orchestrate the data processing and graph creation.
    
    Args:
        data_dir: Directory containing data files
        author_file: Filename for author data
        paper_file: Filename for paper data
        citation_file: Filename for citation data
        processed_dir: Directory to save processed files
        force_reprocess: Whether to force reprocessing even if processed files exist
        
    Returns:
        HeteroData graph object
    """
    print(f"Starting data processing. Processed files will be saved/loaded from: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Setup data paths and loader
    data_paths = DataPaths(base_dir=data_dir)
    loader = CitationDataLoader(data_paths)
    
    # Check if processed graph already exists
    graph_path = os.path.join(processed_dir, 'hetero_graph.pt')
    if os.path.exists(graph_path) and not force_reprocess:
        print(f"Loading existing processed graph from {graph_path}")
        return torch.load(graph_path)
    
    # Load raw data
    try:
        # Raw data should be in the data/raw directory
        raw_dir = data_paths.get_path('raw')
        
        # Author data
        author_path = os.path.join(raw_dir, author_file)
        if not os.path.exists(author_path):
            raise FileNotFoundError(f"Author file not found at {author_path}")
        unique_auids_df = pd.read_csv(author_path)
        
        # Paper data
        paper_path = os.path.join(raw_dir, paper_file)
        if not os.path.exists(paper_path):
            raise FileNotFoundError(f"Paper file not found at {paper_path}")
        merged_df = pd.read_csv(paper_path)
        
        # Citation data
        citation_path = os.path.join(raw_dir, citation_file)
        if not os.path.exists(citation_path):
            raise FileNotFoundError(f"Citation file not found at {citation_path}")
        with open(citation_path, 'r') as f:
            citation_data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Process authors
    author_features, auid_to_idx = preprocess_authors(
        unique_auids_df, save_dir=processed_dir
    )
    
    # Process papers
    paper_features, pmid_to_idx, paper_rcr = preprocess_papers(
        merged_df, pmid_col='pmid', rcr_key='relative_citation_ratio', save_dir=processed_dir
    )
    
    # Build paper-author dictionary
    pmid_auid_dict = loader.build_paper_author_dict(
        unique_auids_df, merged_df, 
        paper_id_col='pmid', author_id_col='AUID'
    )
    
    # Process citation dictionary
    filtered_dict = loader.process_citation_dict(
        citation_data, merged_df, paper_id_col='pmid'
    )
    
    # Create graph
    graph = create_heterogeneous_graph(
        author_features, paper_features,
        auid_to_idx, pmid_to_idx,
        pmid_auid_dict, filtered_dict,
        paper_rcr=paper_rcr,
        save_dir=processed_dir
    )
    
    print("\nGraph Construction Summary:")
    print(graph)
    print("\nData processing and graph creation complete!")
    
    return graph


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process citation data to create a heterogeneous graph")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--output-dir", type=str, default="./processed_graph_data", help="Directory to save processed files")
    parser.add_argument("--author-file", type=str, default="merged_auids.csv", help="Filename for author data")
    parser.add_argument("--paper-file", type=str, default="merged_df_cleaned.csv", help="Filename for paper data")
    parser.add_argument("--citation-file", type=str, default="citing_articles_cleaned.json", help="Filename for citation data")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if processed files exist")
    
    args = parser.parse_args()
    
    # Process data
    graph = process_data_to_graph(
        data_dir=args.data_dir,
        author_file=args.author_file,
        paper_file=args.paper_file,
        citation_file=args.citation_file,
        processed_dir=args.output_dir,
        force_reprocess=args.force
    )
    
    if graph is not None:
        print("\nFinal Graph Summary:")
        print(f"Number of authors: {graph['author'].num_nodes}")
        print(f"Number of papers: {graph['paper'].num_nodes}")
        
        if ('author', 'writes', 'paper') in graph.edge_types:
            print(f"Number of authorship edges: {graph['author', 'writes', 'paper'].edge_index.size(1)}")
        
        if ('paper', 'cites', 'paper') in graph.edge_types:
            print(f"Number of citation edges: {graph['paper', 'cites', 'paper'].edge_index.size(1)}")
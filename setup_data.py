"""
Script to set up the data directory structure and organize data files.
"""
import os
import shutil
import argparse

def create_directory_structure(base_dir='./data'):
    """Create the data directory structure."""
    directories = [
        os.path.join(base_dir, 'raw'),
        os.path.join(base_dir, 'interim'),
        os.path.join(base_dir, 'processed'),
        os.path.join(base_dir, 'embeddings', 'text_embeddings'),
        os.path.join(base_dir, 'external'),
        './results/model_predictions',
        './results/evaluation_metrics',
        './results/visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def copy_data_files(src_files, dest_dir='./data/raw'):
    """
    Copy data files to the appropriate destination.
    
    Args:
        src_files: List of source file paths
        dest_dir: Destination directory
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    for src_file in src_files:
        if not os.path.exists(src_file):
            print(f"Warning: Source file not found: {src_file}")
            continue
        
        file_name = os.path.basename(src_file)
        dest_file = os.path.join(dest_dir, file_name)

        if os.path.abspath(src_file) == os.path.abspath(dest_file):
            print(f"Skipped copying {src_file} (source and destination are the same)")
        else:
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

def main():
    parser = argparse.ArgumentParser(description="Set up data directory structure and organize files")
    parser.add_argument("--data-dir", type=str, default="./data", help="Base data directory")
    parser.add_argument("--author-file", type=str, help="Path to author data file (merged_auids.csv)")
    parser.add_argument("--paper-file", type=str, help="Path to paper data file (merged_df_cleaned.csv)")
    parser.add_argument("--citation-file", type=str, help="Path to citation data file (citing_articles_cleaned.json)")
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure(args.data_dir)
    
    # Define the raw data directory
    raw_dir = os.path.join(args.data_dir, 'raw')
    
    # Collect files to copy
    files_to_copy = []
    if args.author_file:
        files_to_copy.append(args.author_file)
    if args.paper_file:
        files_to_copy.append(args.paper_file)
    if args.citation_file:
        files_to_copy.append(args.citation_file)
    
    # Copy files if specified
    if files_to_copy:
        copy_data_files(files_to_copy, raw_dir)
    else:
        print("\nNo data files specified. Assuming you’ve already placed them in:")
        print(f"  - Author data (merged_auids.csv): {raw_dir}")
        print(f"  - Paper data (merged_df_cleaned.csv): {raw_dir}")
        print(f"  - Citation data (citing_articles_cleaned.json): {raw_dir}")
    
    print("\n✅ Data directory setup complete!")
    print("\nNext steps:")
    print("1. Make sure your data files are in the raw data directory")
    print("2. Run the data processor to create the graph:")
    print("   python data_processor.py --data-dir ./data --output-dir ./processed_graph_data")
    print("3. Train your models using the processed graph")

if __name__ == "__main__":
    main()

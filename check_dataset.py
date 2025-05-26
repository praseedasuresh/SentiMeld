"""
Script to check the structure of the IMDB dataset.
"""
import pandas as pd
import os
import sys

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def check_dataset():
    """Check the structure of the IMDB dataset."""
    dataset_path = os.path.join(config.DATASET_DIR, "IMDB Dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    
    try:
        # Read just the first few rows to understand the structure
        df = pd.read_csv(dataset_path, nrows=5)
        print("Dataset columns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        
        # Count total rows
        total_rows = len(pd.read_csv(dataset_path))
        print(f"\nTotal rows in dataset: {total_rows}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values in first 5 rows:")
        print(missing_values)
        
    except Exception as e:
        print(f"Error reading dataset: {e}")

if __name__ == "__main__":
    check_dataset()

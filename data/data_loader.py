"""
Data loading utilities for the SentiMeld project.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys
import logging
from pathlib import Path

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing of datasets for sentiment analysis.
    Supports multiple dataset types including IMDB reviews and Twitter sentiment data.
    """
    
    def __init__(self, dataset_name: str = "imdb", custom_path: Optional[str] = None):
        """
        Initialize the DataLoader with the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to load ('imdb', 'twitter', or 'custom')
            custom_path: Path to a custom dataset (required if dataset_name is 'custom')
        """
        self.dataset_name = dataset_name
        self.custom_path = custom_path
        
        if dataset_name == "custom" and not custom_path:
            raise ValueError("Must provide custom_path when dataset_name is 'custom'")
            
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the specified dataset.
        
        Returns:
            DataFrame containing the loaded data with 'text' and 'label' columns
        """
        if self.dataset_name == "imdb":
            return self._load_imdb()
        elif self.dataset_name == "twitter":
            return self._load_twitter()
        elif self.dataset_name == "custom":
            return self._load_custom()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_imdb(self) -> pd.DataFrame:
        """Load the IMDB dataset from local file or Hugging Face datasets."""
        logger.info("Loading IMDB dataset...")
        
        # First try to load from local dataset directory
        imdb_path = os.path.join(config.DATASET_DIR, "IMDB Dataset.csv")
        if os.path.exists(imdb_path):
            logger.info(f"Loading IMDB dataset from local file: {imdb_path}")
            try:
                df = pd.read_csv(imdb_path)
                
                # Check if the dataset has the expected columns
                if 'review' in df.columns and 'sentiment' in df.columns:
                    # Map text column to standard name
                    df = df.rename(columns={'review': 'text'})
                    
                    # Map sentiment labels to numeric values for consistency
                    sentiment_map = {'positive': 1, 'negative': 0}
                    df['label'] = df['sentiment'].map(sentiment_map)
                    
                    logger.info(f"Loaded IMDB dataset with {len(df)} samples")
                    return df
                else:
                    logger.warning(f"Local IMDB dataset has unexpected columns: {df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error loading local IMDB dataset: {e}")
        
        # If local file doesn't exist or has issues, try Hugging Face
        try:
            logger.info("Attempting to load IMDB dataset from Hugging Face...")
            dataset = load_dataset("imdb")
            
            # Convert to DataFrame
            train_df = pd.DataFrame({
                'text': dataset['train']['text'],
                'label': dataset['train']['label']
            })
            
            test_df = pd.DataFrame({
                'text': dataset['test']['text'],
                'label': dataset['test']['label']
            })
            
            # Combine train and test
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Map labels (0=negative, 1=positive)
            df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})
            
            logger.info(f"Loaded IMDB dataset from Hugging Face with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading IMDB dataset from Hugging Face: {e}")
            
            # Try alternative paths
            alt_paths = [
                os.path.join(config.RAW_DATA_DIR, "IMDB Dataset.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "IMDB Dataset.csv")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    logger.info(f"Loading IMDB dataset from alternative path: {path}")
                    try:
                        df = pd.read_csv(path)
                        
                        # Check if the dataset has the expected columns
                        if 'review' in df.columns and 'sentiment' in df.columns:
                            # Map text column to standard name
                            df = df.rename(columns={'review': 'text'})
                            
                            # Map sentiment labels to numeric values for consistency
                            sentiment_map = {'positive': 1, 'negative': 0}
                            df['label'] = df['sentiment'].map(sentiment_map)
                            
                            return df
                    except Exception as nested_e:
                        logger.error(f"Error loading IMDB dataset from alternative path: {nested_e}")
            
            raise RuntimeError("Failed to load IMDB dataset from any location")
    
    def _load_twitter(self) -> pd.DataFrame:
        """Load the Twitter sentiment dataset from Hugging Face datasets."""
        logger.info("Loading Twitter sentiment dataset...")
        
        try:
            # Try to load from Hugging Face
            dataset = load_dataset("cardiffnlp/twitter-sentiment")
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'text': dataset['train']['text'],
                'label': dataset['train']['label']
            })
            
            # Map labels (0=negative, 1=neutral, 2=positive)
            df['sentiment'] = df['label'].map({
                0: 'negative', 
                1: 'neutral', 
                2: 'positive'
            })
            
            logger.info(f"Loaded Twitter dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Twitter dataset: {e}")
            
            # Fallback: check if we have a local copy
            twitter_path = os.path.join(config.RAW_DATA_DIR, "twitter_sentiment.csv")
            if os.path.exists(twitter_path):
                logger.info("Loading Twitter dataset from local file...")
                return pd.read_csv(twitter_path)
            else:
                # Try alternative path
                alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "dataset", "twitter_sentiment.csv")
                if os.path.exists(alt_path):
                    logger.info(f"Loading Twitter dataset from alternative path: {alt_path}")
                    return pd.read_csv(alt_path)
                else:
                    raise RuntimeError("Failed to load Twitter dataset and no local copy found")
    
    def _load_custom(self) -> pd.DataFrame:
        """Load a custom dataset from a specified path."""
        if not self.custom_path:
            raise ValueError("custom_path must be provided for custom datasets")
            
        logger.info(f"Loading custom dataset from {self.custom_path}...")
        
        file_ext = os.path.splitext(self.custom_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(self.custom_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.custom_path)
        elif file_ext == '.json':
            df = pd.read_json(self.custom_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Verify required columns
        required_cols = ['text']
        if not all(col in df.columns for col in required_cols):
            # Try to map common column names
            column_mapping = {
                'review': 'text',
                'content': 'text',
                'tweet': 'text',
                'sentence': 'text'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    break
            
            # Check again after mapping
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Custom dataset must contain columns: {required_cols}")
        
        # If no label column, assume we're in inference mode
        if 'label' not in df.columns and 'sentiment' not in df.columns:
            logger.warning("No label/sentiment column found in custom dataset")
            df['label'] = -1  # Placeholder for unlabeled data
            df['sentiment'] = 'unknown'
            
        # If we have label but no sentiment, map numeric labels to text
        if 'label' in df.columns and 'sentiment' not in df.columns:
            # Try to infer the label mapping
            unique_labels = df['label'].unique()
            
            if set(unique_labels) == {0, 1}:
                # Binary sentiment
                df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})
            elif set(unique_labels) == {0, 1, 2}:
                # Three-class sentiment
                df['sentiment'] = df['label'].map({
                    0: 'negative', 
                    1: 'neutral', 
                    2: 'positive'
                })
            else:
                logger.warning(f"Unknown label scheme: {unique_labels}")
                df['sentiment'] = df['label'].astype(str)
        
        # If we have sentiment but no label, map text labels to numeric values
        if 'sentiment' in df.columns and 'label' not in df.columns:
            # Try to infer the sentiment mapping
            unique_sentiments = df['sentiment'].unique()
            
            if set(unique_sentiments) == {'negative', 'positive'}:
                # Binary sentiment
                df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
            elif set(unique_sentiments) == {'negative', 'neutral', 'positive'}:
                # Three-class sentiment
                df['label'] = df['sentiment'].map({
                    'negative': 0, 
                    'neutral': 1, 
                    'positive': 2
                })
            else:
                logger.warning(f"Unknown sentiment scheme: {unique_sentiments}")
                # Create a mapping from unique sentiments to numeric values
                sentiment_to_label = {sent: i for i, sent in enumerate(unique_sentiments)}
                df['label'] = df['sentiment'].map(sentiment_to_label)
        
        logger.info(f"Loaded custom dataset with {len(df)} samples")
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = config.TEST_SIZE, 
                  val_size: float = config.VAL_SIZE, random_state: int = config.RANDOM_SEED
                  ) -> Dict[str, pd.DataFrame]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            df: DataFrame containing the data
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing 'train', 'val', and 'test' DataFrames
        """
        # First split: train + val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label'] if 'label' in df.columns else None
        )
        
        # Second split: train vs val
        # Adjust val_size to account for the reduced size of train_val
        adjusted_val_size = val_size / (1 - test_size)
        
        train, val = train_test_split(
            train_val, test_size=adjusted_val_size, random_state=random_state, 
            stratify=train_val['label'] if 'label' in train_val.columns else None
        )
        
        logger.info(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return {
            'train': train,
            'val': val,
            'test': test
        }
    
    def save_processed_data(self, data_splits: Dict[str, pd.DataFrame]) -> None:
        """
        Save the processed data splits to disk.
        
        Args:
            data_splits: Dictionary containing 'train', 'val', and 'test' DataFrames
        """
        for split_name, df in data_splits.items():
            output_path = os.path.join(config.PROCESSED_DATA_DIR, f"{self.dataset_name}_{split_name}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} data to {output_path}")
    
    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(df),
            'avg_text_length': df['text'].str.len().mean(),
            'median_text_length': df['text'].str.len().median(),
            'max_text_length': df['text'].str.len().max(),
            'min_text_length': df['text'].str.len().min(),
        }
        
        # Add label distribution if we have labels
        if 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            stats['label_distribution'] = label_counts
            
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts().to_dict()
            stats['sentiment_distribution'] = sentiment_counts
            
        return stats
    
    def process_and_prepare_data(self) -> Dict[str, Any]:
        """
        Load, process, split, and save the data.
        
        Returns:
            Dictionary containing data splits and statistics
        """
        # Load the data
        df = self.load_data()
        
        # Get statistics
        stats = self.get_data_stats(df)
        
        # Split the data
        data_splits = self.split_data(df)
        
        # Save processed data
        self.save_processed_data(data_splits)
        
        return {
            'data_splits': data_splits,
            'stats': stats
        }


# Example usage
if __name__ == "__main__":
    # Load IMDB dataset
    imdb_loader = DataLoader(dataset_name="imdb")
    imdb_data = imdb_loader.process_and_prepare_data()
    
    # Print some statistics
    print("IMDB Dataset Statistics:")
    for key, value in imdb_data['stats'].items():
        print(f"{key}: {value}")

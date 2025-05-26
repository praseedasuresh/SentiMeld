"""
Test script to verify the integration of the IMDB dataset and ensure the project runs without errors.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.data_loader import DataLoader
from preprocessing.cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.traditional import TraditionalModel

def test_data_loading():
    """Test loading the IMDB dataset."""
    logger.info("Testing data loading...")
    
    # Initialize data loader
    data_loader = DataLoader(dataset_name="imdb")
    
    try:
        # Load data
        df = data_loader.load_data()
        
        # Check if data was loaded successfully
        if df is not None and len(df) > 0:
            logger.info(f"Successfully loaded IMDB dataset with {len(df)} samples")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"First few rows:\n{df.head()}")
            
            # Check for required columns
            required_cols = ['text', 'label', 'sentiment']
            if all(col in df.columns for col in required_cols):
                logger.info("All required columns are present")
            else:
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"Missing required columns: {missing}")
                
            # Check label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info(f"Label distribution:\n{label_counts}")
                
            # Check sentiment distribution
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                logger.info(f"Sentiment distribution:\n{sentiment_counts}")
                
            return df
        else:
            logger.error("Failed to load data: DataFrame is empty or None")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def test_preprocessing(df):
    """Test preprocessing the IMDB dataset."""
    if df is None:
        logger.error("Cannot test preprocessing: No data provided")
        return None
    
    logger.info("Testing preprocessing...")
    
    try:
        # Initialize text cleaner
        cleaner = TextCleaner()
        
        # Clean a sample of texts
        sample_size = min(100, len(df))
        sample_texts = df['text'].head(sample_size).tolist()
        
        # Time the cleaning process
        import time
        start_time = time.time()
        
        cleaned_texts = cleaner.clean_texts(sample_texts)
        
        end_time = time.time()
        logger.info(f"Cleaned {sample_size} texts in {end_time - start_time:.2f} seconds")
        
        # Check if cleaning was successful
        if cleaned_texts and len(cleaned_texts) == sample_size:
            logger.info("Text cleaning successful")
            
            # Show sample before and after
            for i in range(min(3, sample_size)):
                logger.info(f"Original: {sample_texts[i][:100]}...")
                logger.info(f"Cleaned: {cleaned_texts[i][:100]}...")
            
            return cleaned_texts
        else:
            logger.error("Text cleaning failed")
            return None
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

def test_feature_extraction(cleaned_texts):
    """Test feature extraction on cleaned texts."""
    if cleaned_texts is None:
        logger.error("Cannot test feature extraction: No cleaned texts provided")
        return None
    
    logger.info("Testing feature extraction...")
    
    try:
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(method="tfidf")
        
        # Extract features
        features = feature_extractor.extract_features(cleaned_texts)
        
        # Check if feature extraction was successful
        if features and len(features) > 0:
            logger.info("Feature extraction successful")
            
            # Show feature dimensions
            for feature_name, feature_matrix in features.items():
                if hasattr(feature_matrix, 'shape'):
                    logger.info(f"Feature '{feature_name}' shape: {feature_matrix.shape}")
                elif isinstance(feature_matrix, dict) and all(hasattr(v, 'shape') for v in feature_matrix.values()):
                    for k, v in feature_matrix.items():
                        logger.info(f"Feature '{feature_name}.{k}' shape: {v.shape}")
            
            return features
        else:
            logger.error("Feature extraction failed")
            return None
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return None

def test_model_training(df, features):
    """Test training a simple model on the extracted features."""
    if df is None or features is None:
        logger.error("Cannot test model training: No data or features provided")
        return None
    
    logger.info("Testing model training...")
    
    try:
        # Get a small sample for quick testing
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)
        
        # Get labels
        labels = sample_df['label'].values
        
        # Initialize a simple model
        model = TraditionalModel(model_type="naive_bayes")
        
        # Train the model
        model.fit(features, labels)
        
        logger.info("Model training successful")
        
        # Make predictions
        predictions = model.predict(features)
        
        # Check predictions
        if predictions is not None and len(predictions) == sample_size:
            logger.info("Model prediction successful")
            
            # Calculate accuracy
            accuracy = np.mean(predictions == labels)
            logger.info(f"Accuracy on training data: {accuracy:.4f}")
            
            return model
        else:
            logger.error("Model prediction failed")
            return None
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return None

def main():
    """Main function to run all tests."""
    logger.info("Starting integration tests...")
    
    # Test data loading
    df = test_data_loading()
    
    if df is not None:
        # Test preprocessing
        cleaned_texts = test_preprocessing(df)
        
        if cleaned_texts is not None:
            # Test feature extraction
            features = test_feature_extraction(cleaned_texts)
            
            if features is not None:
                # Test model training
                model = test_model_training(df, features)
                
                if model is not None:
                    logger.info("All tests passed successfully!")
                    return True
    
    logger.error("Integration tests failed")
    return False

if __name__ == "__main__":
    main()

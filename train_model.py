"""
Script to train a basic model and save the necessary files for the SentiMeld project.
This ensures that the analyze mode works properly without errors.
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
from models.deep_learning import TransformerModel
from models.ensemble import EnsembleModel

def train_and_save_model():
    """Train a basic model and save the necessary files."""
    logger.info("Training a basic model for sentiment analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.MODEL_ARTIFACTS_DIR, exist_ok=True)
    
    # Step 1: Load data
    logger.info("Loading IMDB dataset...")
    data_loader = DataLoader(dataset_name="imdb")
    try:
        df = data_loader.load_data()
        logger.info(f"Loaded {len(df)} samples from IMDB dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False
    
    # Step 2: Process and split data
    logger.info("Processing and splitting data...")
    try:
        # Split data into train/val/test
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return False
    
    # Step 3: Clean text
    logger.info("Cleaning text data...")
    cleaner = TextCleaner()
    try:
        # Use a smaller subset for faster training
        sample_size = min(5000, len(train_df))
        train_sample = train_df.iloc[:sample_size]
        
        train_texts = cleaner.clean_texts(train_sample['text'].tolist())
        test_texts = cleaner.clean_texts(test_df['text'].head(1000).tolist())
        
        logger.info(f"Cleaned {len(train_texts)} training texts and {len(test_texts)} test texts")
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return False
    
    # Step 4: Extract features
    logger.info("Extracting features...")
    feature_extractor = FeatureExtractor(method="tfidf")
    try:
        train_features = feature_extractor.extract_features(train_texts, fit=True)
        test_features = feature_extractor.extract_features(test_texts, fit=False)
        
        logger.info("Feature extraction completed")
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return False
    
    # Step 5: Train traditional model
    logger.info("Training traditional model...")
    traditional_model = TraditionalModel(model_type="logistic_regression")
    try:
        train_labels = train_sample['label'].values
        traditional_model.fit(train_features, train_labels)
        
        # Evaluate on test set
        test_labels = test_df['label'].head(1000).values
        metrics = traditional_model.evaluate(test_features, test_labels)
        
        logger.info(f"Traditional model trained. Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error training traditional model: {e}")
        return False
    
    # Step 6: Save models and feature extractor
    logger.info("Saving models and feature extractor...")
    try:
        # Save traditional model
        traditional_model_path = os.path.join(config.MODEL_ARTIFACTS_DIR, "sentimeld_model_traditional.pkl")
        traditional_model.save(traditional_model_path)
        logger.info(f"Traditional model saved to {traditional_model_path}")
        
        # Save feature extractor
        feature_extractor_path = os.path.join(config.MODEL_ARTIFACTS_DIR, "sentimeld_model_feature_extractor.pkl")
        feature_extractor.save(feature_extractor_path)
        logger.info(f"Feature extractor saved to {feature_extractor_path}")
        
        # Create a simple ensemble model (just using the traditional model for now)
        ensemble_model = EnsembleModel()
        ensemble_model.add_model("traditional", traditional_model)
        
        # Save ensemble model
        ensemble_model_path = os.path.join(config.MODEL_ARTIFACTS_DIR, "sentimeld_model_ensemble.pkl")
        ensemble_model.save(ensemble_model_path)
        logger.info(f"Ensemble model saved to {ensemble_model_path}")
        
        logger.info("All models and feature extractor saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training process...")
    success = train_and_save_model()
    
    if success:
        logger.info("Model training and saving completed successfully!")
        logger.info("You can now use the analyze mode with the command:")
        logger.info('python main.py --mode analyze --model ensemble --text "Your text here"')
    else:
        logger.error("Model training and saving failed. Please check the logs for details.")

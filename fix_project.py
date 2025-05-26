"""
Script to fix the SentiMeld project issues and ensure it runs properly.
This script will train and save models with the correct file paths.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
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
from models.ensemble import EnsembleModel

def fix_project():
    """Fix the SentiMeld project issues."""
    logger.info("Starting to fix the SentiMeld project...")
    
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
        
        # Ensure train_features and test_features are dictionaries
        if not isinstance(train_features, dict):
            train_features = {'features': train_features}
        if not isinstance(test_features, dict):
            test_features = {'features': test_features}
            
        logger.info("Feature extraction completed")
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return False
    
    # Step 5: Train traditional model
    logger.info("Training traditional model...")
    traditional_model = TraditionalModel(model_type="logistic_regression")
    try:
        train_labels = train_sample['label'].values
        
        # Ensure we're passing the right format of features
        if 'tfidf_features' in train_features:
            # If using the dictionary format directly
            traditional_model.fit(train_features, train_labels)
        else:
            # If we had to wrap it in a dictionary
            traditional_model.fit({'features': train_features['features']}, train_labels)
        
        # Evaluate on test set
        test_labels = test_df['label'].head(1000).values
        
        # Use the same format for test features
        if 'tfidf_features' in test_features:
            metrics = traditional_model.evaluate(test_features, test_labels)
        else:
            metrics = traditional_model.evaluate({'features': test_features['features']}, test_labels)
        
        logger.info(f"Traditional model trained. Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error training traditional model: {e}")
        return False
    
    # Step 6: Save models and feature extractor
    logger.info("Saving models and feature extractor...")
    model_name = "sentimeld_model"
    try:
        # Save feature extractor
        feature_extractor_path = os.path.join(config.MODEL_ARTIFACTS_DIR, f"{model_name}_feature_extractor.pkl")
        feature_extractor.save(feature_extractor_path)
        logger.info(f"Feature extractor saved to {feature_extractor_path}")
        
        # Save traditional model with the correct name
        traditional_model_path = os.path.join(config.MODEL_ARTIFACTS_DIR, f"{model_name}_traditional.pkl")
        traditional_model.save(traditional_model_path)
        logger.info(f"Traditional model saved to {traditional_model_path}")
        
        # Create and save ensemble model
        ensemble_model = EnsembleModel()
        ensemble_model.add_model("traditional", traditional_model)
        
        # Set classes for the ensemble model
        ensemble_model.classes_ = traditional_model.classes_
        
        # Save ensemble model
        ensemble_model_path = os.path.join(config.MODEL_ARTIFACTS_DIR, f"{model_name}_ensemble.pkl")
        
        # Save ensemble data
        ensemble_data = {
            'weights': {'traditional': 1.0},
            'classes_': ensemble_model.classes_
        }
        
        with open(ensemble_model_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save individual models with correct names
        model_dir = os.path.dirname(ensemble_model_path)
        
        # Save traditional model again with the name expected by the ensemble loader
        traditional_path = os.path.join(model_dir, f"{model_name}_traditional.pkl")
        if not os.path.exists(traditional_path):
            traditional_model.save(traditional_path)
        
        logger.info(f"Ensemble model saved to {ensemble_model_path}")
        logger.info("All models and feature extractor saved successfully")
        
        # Create a file to indicate the model paths
        with open(os.path.join(config.MODEL_ARTIFACTS_DIR, "model_paths.txt"), "w") as f:
            f.write(f"Feature Extractor: {feature_extractor_path}\n")
            f.write(f"Traditional Model: {traditional_model_path}\n")
            f.write(f"Ensemble Model: {ensemble_model_path}\n")
        
        return True
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def clean_project():
    """Remove unwanted files from the project."""
    logger.info("Cleaning up unwanted files...")
    
    # List of files to remove
    unwanted_files = [
        os.path.join(config.MODEL_ARTIFACTS_DIR, "transformer.pkl"),
        os.path.join(config.MODEL_ARTIFACTS_DIR, "traditional.pkl")
    ]
    
    for file_path in unwanted_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed unwanted file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")

if __name__ == "__main__":
    logger.info("Starting project fix process...")
    
    # Clean up unwanted files
    clean_project()
    
    # Fix the project
    success = fix_project()
    
    if success:
        logger.info("Project fix completed successfully!")
        logger.info("You can now use the analyze mode with the command:")
        logger.info('python main.py --mode analyze --model ensemble --text "Your text here"')
    else:
        logger.error("Project fix failed. Please check the logs for details.")

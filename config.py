"""
Configuration settings for the SentiMeld project.
"""
import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACTS_DIR = ROOT_DIR / "model_artifacts"
DATASET_DIR = ROOT_DIR / "dataset"  # Add dataset directory

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MODEL_ARTIFACTS_DIR, DATASET_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Data settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Text preprocessing settings
MAX_SEQUENCE_LENGTH = 128
MIN_WORD_FREQUENCY = 3
MAX_VOCAB_SIZE = 50000
STOPWORDS_EXCLUDE = ['not', 'no', 'never', 'nor', 'neither', 'hardly', 'scarcely']

# Model settings
TRADITIONAL_MODELS = {
    'naive_bayes': {
        'alpha': 1.0,
    },
    'svm': {
        'C': 1.0,
        'kernel': 'linear',
    },
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
    }
}

TRANSFORMER_MODEL = 'distilbert-base-uncased'
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_EPOCHS = 3

# Ensemble settings
ENSEMBLE_WEIGHTS = {
    'traditional': 0.3,
    'transformer': 0.7
}

# Aspect-based settings
ASPECTS = {
    'product_reviews': ['price', 'quality', 'service', 'shipping', 'design', 'functionality'],
    'tweets': ['relevance', 'sentiment', 'emotion', 'stance']
}

# Emotion categories
EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Dashboard settings
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

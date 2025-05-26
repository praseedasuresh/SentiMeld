"""
Setup script for the SentiMeld project.
This script installs required packages and downloads necessary data files.
"""
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages from requirements.txt."""
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Successfully installed required packages.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install required packages: {e}")
        sys.exit(1)

def download_nltk_data():
    """Download required NLTK data."""
    logger.info("Downloading NLTK data...")
    import nltk
    
    nltk_resources = [
        'punkt',
        'stopwords',
        'wordnet'
    ]
    
    for resource in nltk_resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=False)
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {e}")

def download_spacy_model():
    """Download required SpaCy model."""
    logger.info("Downloading SpaCy model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("Successfully downloaded SpaCy model.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download SpaCy model: {e}")

def create_directories():
    """Create necessary directories."""
    logger.info("Creating necessary directories...")
    directories = [
        "data/raw",
        "data/processed",
        "model_artifacts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main function."""
    logger.info("Setting up SentiMeld project...")
    
    # Install required packages
    install_requirements()
    
    # Download NLTK data
    download_nltk_data()
    
    # Download SpaCy model
    download_spacy_model()
    
    # Create directories
    create_directories()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the SentiMeld tool using:")
    logger.info("python main.py --mode train --dataset imdb --model ensemble")
    logger.info("or")
    logger.info("python main.py --mode analyze --model ensemble --text \"Your text here\"")

if __name__ == "__main__":
    main()

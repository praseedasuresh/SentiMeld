"""
Feature extraction utilities for the SentiMeld project.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from transformers import AutoTokenizer
import sys
import os
import logging
import pickle
from collections import Counter

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Class for extracting features from text data for sentiment analysis.
    Supports multiple feature extraction methods:
    - Bag of Words (BoW)
    - TF-IDF
    - N-grams
    - Transformer embeddings
    - Topic modeling features
    - Lexical features
    """
    
    def __init__(self, method: str = "tfidf", 
                 max_features: int = config.MAX_VOCAB_SIZE,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = config.MIN_WORD_FREQUENCY,
                 transformer_model: str = config.TRANSFORMER_MODEL,
                 use_topic_features: bool = False,
                 n_topics: int = 10,
                 use_lexical_features: bool = False):
        """
        Initialize the FeatureExtractor with the specified options.
        
        Args:
            method: Feature extraction method ('bow', 'tfidf', 'transformer', 'ensemble')
            max_features: Maximum number of features for BoW and TF-IDF
            ngram_range: Range of n-grams to consider for BoW and TF-IDF
            min_df: Minimum document frequency for BoW and TF-IDF
            transformer_model: Transformer model to use for embeddings
            use_topic_features: Whether to include topic modeling features
            n_topics: Number of topics for topic modeling
            use_lexical_features: Whether to include lexical features
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.transformer_model = transformer_model
        self.use_topic_features = use_topic_features
        self.n_topics = n_topics
        self.use_lexical_features = use_lexical_features
        
        # Initialize vectorizers
        if method in ['bow', 'ensemble']:
            self.bow_vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df
            )
        
        if method in ['tfidf', 'ensemble']:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df
            )
        
        if method in ['transformer', 'ensemble']:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        
        # Initialize topic modeling
        if use_topic_features:
            self.lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=config.RANDOM_SEED
            )
            self.svd = TruncatedSVD(
                n_components=n_topics,
                random_state=config.RANDOM_SEED
            )
        
        # Lexical features
        if use_lexical_features:
            # Load sentiment lexicons
            self.load_lexicons()
    
    def load_lexicons(self):
        """Load sentiment lexicons for lexical feature extraction."""
        # This is a simplified version - in a real implementation, you would load actual lexicons
        # For now, we'll use some example positive and negative words
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'terrific', 'outstanding', 'superb', 'brilliant', 'awesome',
            'love', 'like', 'enjoy', 'recommend', 'positive', 'best',
            'perfect', 'happy', 'pleased', 'satisfied', 'impressive'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing',
            'worst', 'hate', 'dislike', 'negative', 'mediocre', 'boring',
            'waste', 'annoying', 'frustrating', 'useless', 'unpleasant',
            'sad', 'angry', 'upset', 'dissatisfied', 'regret'
        }
    
    def extract_bow_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract Bag of Words features from texts.
        
        Args:
            texts: List of input texts
            fit: Whether to fit the vectorizer on the data
            
        Returns:
            Bag of Words feature matrix
        """
        if fit:
            return self.bow_vectorizer.fit_transform(texts)
        else:
            return self.bow_vectorizer.transform(texts)
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: List of input texts
            fit: Whether to fit the vectorizer on the data
            
        Returns:
            TF-IDF feature matrix
        """
        if fit:
            return self.tfidf_vectorizer.fit_transform(texts)
        else:
            return self.tfidf_vectorizer.transform(texts)
    
    def extract_transformer_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract transformer features (token IDs, attention mask) from texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with transformer features
        """
        # Tokenize the texts
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=config.MAX_SEQUENCE_LENGTH,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def extract_topic_features(self, texts: List[str], fit: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract topic modeling features from texts.
        
        Args:
            texts: List of input texts
            fit: Whether to fit the topic models on the data
            
        Returns:
            Dictionary with topic modeling features
        """
        # First, get TF-IDF features
        if hasattr(self, 'tfidf_vectorizer'):
            if fit:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        else:
            # Create a temporary TF-IDF vectorizer
            temp_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df
            )
            if fit:
                tfidf_matrix = temp_vectorizer.fit_transform(texts)
            else:
                tfidf_matrix = temp_vectorizer.transform(texts)
        
        # Extract LDA features
        if fit:
            lda_features = self.lda.fit_transform(tfidf_matrix)
        else:
            lda_features = self.lda.transform(tfidf_matrix)
        
        # Extract SVD features
        if fit:
            svd_features = self.svd.fit_transform(tfidf_matrix)
        else:
            svd_features = self.svd.transform(tfidf_matrix)
        
        return {
            'lda_features': lda_features,
            'svd_features': svd_features
        }
    
    def extract_lexical_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract lexical features from texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Lexical feature matrix
        """
        features = []
        
        for text in texts:
            # Tokenize the text
            words = text.lower().split()
            
            # Count positive and negative words
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            
            # Calculate other lexical features
            word_count = len(words)
            unique_word_count = len(set(words))
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Calculate lexical diversity
            lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
            
            # Calculate sentiment polarity
            polarity = (positive_count - negative_count) / (positive_count + negative_count + 1e-10)
            
            # Calculate sentiment intensity
            intensity = (positive_count + negative_count) / (word_count + 1e-10)
            
            # Create feature vector
            feature_vector = [
                positive_count,
                negative_count,
                positive_count / (word_count + 1e-10),
                negative_count / (word_count + 1e-10),
                polarity,
                intensity,
                word_count,
                unique_word_count,
                lexical_diversity,
                avg_word_length
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_features(self, texts: List[str], fit: bool = True) -> Dict[str, Any]:
        """
        Extract features from texts using the specified method.
        
        Args:
            texts: List of input texts
            fit: Whether to fit the feature extractors on the data
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        if self.method in ['bow', 'ensemble']:
            features['bow_features'] = self.extract_bow_features(texts, fit)
        
        if self.method in ['tfidf', 'ensemble']:
            features['tfidf_features'] = self.extract_tfidf_features(texts, fit)
        
        if self.method in ['transformer', 'ensemble']:
            transformer_features = self.extract_transformer_features(texts)
            features.update(transformer_features)
        
        if self.use_topic_features:
            topic_features = self.extract_topic_features(texts, fit)
            features.update(topic_features)
        
        if self.use_lexical_features:
            features['lexical_features'] = self.extract_lexical_features(texts)
        
        return features
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the names of the features.
        
        Returns:
            Dictionary with feature names
        """
        feature_names = {}
        
        if hasattr(self, 'bow_vectorizer'):
            feature_names['bow_features'] = self.bow_vectorizer.get_feature_names_out().tolist()
        
        if hasattr(self, 'tfidf_vectorizer'):
            feature_names['tfidf_features'] = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        if self.use_topic_features:
            feature_names['lda_features'] = [f'topic_{i}' for i in range(self.n_topics)]
            feature_names['svd_features'] = [f'svd_topic_{i}' for i in range(self.n_topics)]
        
        if self.use_lexical_features:
            feature_names['lexical_features'] = [
                'positive_count',
                'negative_count',
                'positive_ratio',
                'negative_ratio',
                'polarity',
                'intensity',
                'word_count',
                'unique_word_count',
                'lexical_diversity',
                'avg_word_length'
            ]
        
        return feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save the feature extractor to disk.
        
        Args:
            filepath: Path to save the feature extractor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a feature extractor from disk.
        
        Args:
            filepath: Path to load the feature extractor from
            
        Returns:
            Loaded feature extractor
        """
        with open(filepath, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        logger.info(f"Feature extractor loaded from {filepath}")
        return feature_extractor


# Example usage
if __name__ == "__main__":
    # Example texts
    texts = [
        "This movie was amazing! I loved it so much.",
        "The product was terrible. Would not recommend.",
        "The service was okay, but could be better."
    ]
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        method="ensemble",
        use_topic_features=True,
        use_lexical_features=True
    )
    
    # Extract features
    features = feature_extractor.extract_features(texts)
    
    # Print feature shapes
    for name, feature in features.items():
        if isinstance(feature, np.ndarray):
            print(f"{name}: {feature.shape}")
        else:
            print(f"{name}: {type(feature)}")

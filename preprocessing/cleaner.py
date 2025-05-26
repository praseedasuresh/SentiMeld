"""
Text cleaning utilities for the SentiMeld project.
"""
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
import sys
import os
import logging
from typing import List, Set, Dict, Any, Optional

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextCleaner:
    """
    Class for cleaning and preprocessing text data for sentiment analysis.
    """
    
    def __init__(self, remove_stopwords: bool = True, 
                 lemmatize: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True,
                 remove_urls: bool = True,
                 remove_html_tags: bool = True,
                 expand_contractions: bool = True,
                 lowercase: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 keep_negation_stopwords: bool = True):
        """
        Initialize the TextCleaner with the specified options.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_urls: Whether to remove URLs
            remove_html_tags: Whether to remove HTML tags
            expand_contractions: Whether to expand contractions (e.g., "don't" -> "do not")
            lowercase: Whether to convert text to lowercase
            custom_stopwords: Additional stopwords to remove
            keep_negation_stopwords: Whether to keep negation words in stopwords removal
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_html_tags = remove_html_tags
        self.expand_contractions = expand_contractions
        self.lowercase = lowercase
        
        # Initialize lemmatizer if needed
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stopwords if needed
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
            
            # Keep negation words if specified
            if keep_negation_stopwords:
                self.stopwords -= set(config.STOPWORDS_EXCLUDE)
            
            # Add custom stopwords if provided
            if custom_stopwords:
                self.stopwords.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text: The input text to clean
            
        Returns:
            The cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags if specified
        if self.remove_html_tags:
            text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs if specified
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Expand contractions if specified
        if self.expand_contractions:
            text = contractions.fix(text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
        
        # Lemmatize if specified
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean and preprocess a list of texts.
        
        Args:
            texts: List of input texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def get_cleaning_stats(self, original_texts: List[str], cleaned_texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process.
        
        Args:
            original_texts: List of original texts
            cleaned_texts: List of cleaned texts
            
        Returns:
            Dictionary with cleaning statistics
        """
        original_lengths = [len(text) for text in original_texts]
        cleaned_lengths = [len(text) for text in cleaned_texts]
        
        original_word_counts = [len(text.split()) for text in original_texts]
        cleaned_word_counts = [len(text.split()) for text in cleaned_texts]
        
        stats = {
            'avg_original_length': sum(original_lengths) / len(original_lengths),
            'avg_cleaned_length': sum(cleaned_lengths) / len(cleaned_lengths),
            'avg_original_word_count': sum(original_word_counts) / len(original_word_counts),
            'avg_cleaned_word_count': sum(cleaned_word_counts) / len(cleaned_word_counts),
            'avg_reduction_percentage': 100 * (1 - sum(cleaned_lengths) / sum(original_lengths))
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize the cleaner with default options
    cleaner = TextCleaner()
    
    # Example text
    text = "This movie was AMAZING! I loved it so much. The acting was great, and the plot was interesting. 10/10 would recommend! Check it out at https://example.com. #MustWatch"
    
    # Clean the text
    cleaned_text = cleaner.clean_text(text)
    
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned_text}")

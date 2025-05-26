"""
Traditional machine learning models for sentiment analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys
import os
import logging
import pickle
import joblib
from scipy.sparse import hstack, csr_matrix, issparse

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TraditionalModel:
    """
    Class for traditional machine learning models for sentiment analysis.
    Supports multiple models:
    - Naive Bayes
    - SVM
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    """
    
    def __init__(self, model_type: str = "naive_bayes", model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the model with the specified type and parameters.
        
        Args:
            model_type: Type of model to use ('naive_bayes', 'svm', 'logistic_regression', 'random_forest', 'gradient_boosting')
            model_params: Parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.classes_ = None
        self.feature_importances_ = None
    
    def _create_model(self):
        """
        Create a model of the specified type with the specified parameters.
        
        Returns:
            The created model
        """
        if self.model_type == "naive_bayes":
            params = {**config.TRADITIONAL_MODELS.get('naive_bayes', {}), **self.model_params}
            return MultinomialNB(**params)
        
        elif self.model_type == "svm":
            params = {**config.TRADITIONAL_MODELS.get('svm', {}), **self.model_params}
            return SVC(**params, probability=True)
        
        elif self.model_type == "logistic_regression":
            params = {**config.TRADITIONAL_MODELS.get('logistic_regression', {}), **self.model_params}
            return LogisticRegression(**params)
        
        elif self.model_type == "random_forest":
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': config.RANDOM_SEED,
                **self.model_params
            }
            return RandomForestClassifier(**params)
        
        elif self.model_type == "gradient_boosting":
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': config.RANDOM_SEED,
                **self.model_params
            }
            return GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _combine_features(self, features: Dict[str, Any]) -> Union[np.ndarray, csr_matrix]:
        """
        Combine multiple feature matrices into a single matrix.
        
        Args:
            features: Dictionary of feature matrices
            
        Returns:
            Combined feature matrix
        """
        feature_matrices = []
        
        # Process each feature type
        for name, feature in features.items():
            if name in ['input_ids', 'attention_mask']:
                # Skip transformer features
                continue
            
            if isinstance(feature, np.ndarray):
                # Convert to sparse matrix if it's a dense array
                if len(feature.shape) == 2:
                    feature_matrices.append(csr_matrix(feature))
            elif issparse(feature):
                feature_matrices.append(feature)
        
        if not feature_matrices:
            raise ValueError("No valid features found")
        
        # Combine all feature matrices
        if len(feature_matrices) == 1:
            return feature_matrices[0]
        else:
            return hstack(feature_matrices)
    
    def fit(self, features: Dict[str, Any], labels: np.ndarray) -> 'TraditionalModel':
        """
        Fit the model to the data.
        
        Args:
            features: Dictionary of feature matrices
            labels: Array of labels
            
        Returns:
            Self
        """
        # Combine features
        X = self._combine_features(features)
        
        logger.info(f"Fitting {self.model_type} model to data with shape {X.shape}")
        
        # Fit the model
        self.model.fit(X, labels)
        
        # Store classes
        self.classes_ = self.model.classes_
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            if len(self.model.coef_.shape) == 2:
                # Multi-class case
                self.feature_importances_ = np.abs(self.model.coef_).mean(axis=0)
            else:
                # Binary case
                self.feature_importances_ = np.abs(self.model.coef_)
        
        return self
    
    def predict(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Predict labels for the data.
        
        Args:
            features: Dictionary of feature matrices
            
        Returns:
            Array of predicted labels
        """
        # Combine features
        X = self._combine_features(features)
        
        # Predict
        return self.model.predict(X)
    
    def predict_proba(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Predict class probabilities for the data.
        
        Args:
            features: Dictionary of feature matrices
            
        Returns:
            Array of predicted class probabilities
        """
        # Combine features
        X = self._combine_features(features)
        
        # Predict probabilities
        return self.model.predict_proba(X)
    
    def evaluate(self, features: Dict[str, Any], labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on the data.
        
        Args:
            features: Dictionary of feature matrices
            labels: Array of labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict
        predictions = self.predict(features)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
            'classification_report': classification_report(labels, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        logger.info(f"Evaluation metrics for {self.model_type} model:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get the feature importances.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importances
        """
        if self.feature_importances_ is None:
            logger.warning("Feature importances not available for this model")
            return {}
        
        if len(feature_names) != len(self.feature_importances_):
            logger.warning(f"Feature names length ({len(feature_names)}) does not match "
                          f"feature importances length ({len(self.feature_importances_)})")
            return {}
        
        return dict(zip(feature_names, self.feature_importances_))
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'classes_': self.classes_,
            'feature_importances_': self.feature_importances_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TraditionalModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        instance = cls(model_type=model_data['model_type'], model_params=model_data['model_params'])
        
        # Replace the model
        instance.model = model_data['model']
        instance.classes_ = model_data['classes_']
        instance.feature_importances_ = model_data['feature_importances_']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


# Example usage
if __name__ == "__main__":
    # Example data
    X = {
        'tfidf_features': csr_matrix(np.random.rand(100, 1000)),
        'lexical_features': np.random.rand(100, 10)
    }
    y = np.random.randint(0, 3, 100)
    
    # Initialize model
    model = TraditionalModel(model_type="logistic_regression")
    
    # Fit model
    model.fit(X, y)
    
    # Evaluate model
    metrics = model.evaluate(X, y)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

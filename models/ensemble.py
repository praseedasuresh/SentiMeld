"""
Ensemble model for sentiment analysis that combines multiple models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys
import os
import logging
import pickle
from scipy.special import softmax

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.traditional import TraditionalModel
from models.deep_learning import TransformerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    """Class for ensemble models that combine multiple models for sentiment analysis."""
    
    def __init__(self, models: Optional[Dict[str, Any]] = None, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            models: Dictionary of models to ensemble
            weights: Dictionary of weights for each model
        """
        self.models = models or {}
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        self.classes_ = None
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: The model to add
            weight: Weight of the model in the ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def fit(self, features: Dict[str, Dict[str, Any]], labels: np.ndarray,
           val_features: Optional[Dict[str, Dict[str, Any]]] = None,
           val_labels: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Fit all models in the ensemble.
        
        Args:
            features: Dictionary of features for each model
            labels: Array of labels
            val_features: Optional validation features
            val_labels: Optional validation labels
            
        Returns:
            Dictionary with training history
        """
        history = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            if name not in features:
                logger.warning(f"No features found for {name} model, skipping...")
                continue
            
            # Get features for this model
            model_features = features[name]
            
            # Get validation features if available
            model_val_features = val_features[name] if val_features and name in val_features else None
            
            # Fit model
            if hasattr(model, 'fit'):
                if val_features and val_labels is not None and model_val_features is not None:
                    # If model supports validation data
                    if 'val_features' in model.fit.__code__.co_varnames and 'val_labels' in model.fit.__code__.co_varnames:
                        model_history = model.fit(model_features, labels, val_features=model_val_features, val_labels=val_labels)
                    else:
                        model_history = model.fit(model_features, labels)
                else:
                    model_history = model.fit(model_features, labels)
                
                # Add model history to ensemble history
                if isinstance(model_history, dict):
                    for key, value in model_history.items():
                        history[f"{name}_{key}"] = value
        
        # Store classes
        for name, model in self.models.items():
            if hasattr(model, 'classes_'):
                self.classes_ = model.classes_
                break
        
        return history
    
    def predict(self, features: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Predict labels using the ensemble.
        
        Args:
            features: Dictionary of features for each model
            
        Returns:
            Array of predicted labels
        """
        # Get predictions from each model
        predictions = {}
        
        for name, model in self.models.items():
            if name not in features:
                logger.warning(f"No features found for {name} model, skipping...")
                continue
            
            # Get features for this model
            model_features = features[name]
            
            # Predict
            if hasattr(model, 'predict'):
                predictions[name] = model.predict(model_features)
        
        # If no predictions, return empty array
        if not predictions:
            return np.array([])
        
        # Get shape of predictions
        first_pred = next(iter(predictions.values()))
        n_samples = len(first_pred)
        
        # Initialize weighted predictions
        if self.classes_ is not None:
            n_classes = len(self.classes_)
        else:
            # Try to infer number of classes
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features[name])
                    n_classes = proba.shape[1]
                    break
            else:
                # Default to binary classification
                n_classes = 2
        
        weighted_proba = np.zeros((n_samples, n_classes))
        
        # Add weighted probabilities
        for name, model in self.models.items():
            if name not in features or name not in predictions:
                continue
            
            # Get weight
            weight = self.weights.get(name, 1.0)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features[name])
            else:
                # Convert predictions to one-hot
                pred = predictions[name]
                proba = np.zeros((n_samples, n_classes))
                for i, p in enumerate(pred):
                    proba[i, p] = 1.0
            
            # Add weighted probabilities
            weighted_proba += weight * proba
        
        # Get final predictions
        return np.argmax(weighted_proba, axis=1)
    
    def predict_proba(self, features: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            features: Dictionary of features for each model
            
        Returns:
            Array of predicted class probabilities
        """
        # Get predictions from each model
        probas = {}
        
        for name, model in self.models.items():
            if name not in features:
                logger.warning(f"No features found for {name} model, skipping...")
                continue
            
            # Get features for this model
            model_features = features[name]
            
            # Predict probabilities
            if hasattr(model, 'predict_proba'):
                probas[name] = model.predict_proba(model_features)
            elif hasattr(model, 'predict'):
                # Convert predictions to one-hot
                pred = model.predict(model_features)
                
                # Get number of classes
                if self.classes_ is not None:
                    n_classes = len(self.classes_)
                else:
                    # Try to infer number of classes
                    unique_classes = np.unique(pred)
                    n_classes = max(len(unique_classes), 2)
                
                # Create one-hot encoding
                proba = np.zeros((len(pred), n_classes))
                for i, p in enumerate(pred):
                    proba[i, p] = 1.0
                
                probas[name] = proba
        
        # If no probabilities, return empty array
        if not probas:
            return np.array([])
        
        # Get shape of probabilities
        first_proba = next(iter(probas.values()))
        n_samples, n_classes = first_proba.shape
        
        # Initialize weighted probabilities
        weighted_proba = np.zeros((n_samples, n_classes))
        
        # Add weighted probabilities
        for name, proba in probas.items():
            # Get weight
            weight = self.weights.get(name, 1.0)
            
            # Add weighted probabilities
            weighted_proba += weight * proba
        
        # Normalize probabilities
        row_sums = weighted_proba.sum(axis=1)
        weighted_proba = weighted_proba / row_sums[:, np.newaxis]
        
        return weighted_proba
    
    def evaluate(self, features: Dict[str, Dict[str, Any]], labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the ensemble on the data.
        
        Args:
            features: Dictionary of features for each model
            labels: Array of labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        predictions = self.predict(features)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
            'classification_report': classification_report(labels, predictions),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save the ensemble model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble data
        ensemble_data = {
            'weights': self.weights,
            'classes_': self.classes_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save individual models
        model_dir = os.path.dirname(filepath)
        
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{name}.pkl")
            
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, load_models: bool = True) -> 'EnsembleModel':
        """
        Load an ensemble model from disk.
        
        Args:
            filepath: Path to load the model from
            load_models: Whether to load the individual models
            
        Returns:
            Loaded ensemble model
        """
        # Load ensemble data
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        # Create instance
        instance = cls(weights=ensemble_data['weights'])
        instance.classes_ = ensemble_data['classes_']
        
        # Load individual models
        if load_models:
            model_dir = os.path.dirname(filepath)
            
            for name in ensemble_data['weights'].keys():
                model_path = os.path.join(model_dir, f"{name}.pkl")
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                
                try:
                    if name == 'transformer':
                        # Load transformer model
                        model = TransformerModel.load(model_path)
                    elif name == 'traditional':
                        # Load traditional model
                        model = TraditionalModel.load(model_path)
                    else:
                        # Load generic model
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    
                    instance.models[name] = model
                except Exception as e:
                    logger.error(f"Error loading model {name}: {e}")
        
        logger.info(f"Ensemble model loaded from {filepath}")
        return instance


class AspectBasedSentimentModel:
    """
    Class for aspect-based sentiment analysis.
    """
    
    def __init__(self, aspects: List[str], base_model: Optional[Any] = None):
        """
        Initialize the aspect-based sentiment model.
        
        Args:
            aspects: List of aspects to analyze
            base_model: Base model to use for sentiment analysis
        """
        self.aspects = aspects
        self.base_model = base_model
        self.aspect_models = {}
    
    def fit(self, features: Dict[str, Dict[str, Any]], aspect_labels: Dict[str, np.ndarray]) -> None:
        """
        Fit models for each aspect.
        
        Args:
            features: Dictionary of features for each model
            aspect_labels: Dictionary of labels for each aspect
        """
        for aspect in self.aspects:
            if aspect not in aspect_labels:
                logger.warning(f"No labels found for aspect: {aspect}")
                continue
            
            logger.info(f"Fitting model for aspect: {aspect}")
            
            # Create a copy of the base model
            if self.base_model is not None:
                if hasattr(self.base_model, '__class__'):
                    model = self.base_model.__class__()
                else:
                    model = type(self.base_model)()
            else:
                # Default to a traditional model
                model = TraditionalModel(model_type="logistic_regression")
            
            # Fit the model
            model.fit(features, aspect_labels[aspect])
            
            # Store the model
            self.aspect_models[aspect] = model
    
    def predict(self, features: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Predict sentiment for each aspect.
        
        Args:
            features: Dictionary of features for each model
            
        Returns:
            Dictionary of predicted labels for each aspect
        """
        predictions = {}
        
        for aspect, model in self.aspect_models.items():
            logger.info(f"Predicting for aspect: {aspect}")
            
            # Get predictions
            preds = model.predict(features)
            predictions[aspect] = preds
        
        return predictions
    
    def predict_proba(self, features: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities for each aspect.
        
        Args:
            features: Dictionary of features for each model
            
        Returns:
            Dictionary of predicted class probabilities for each aspect
        """
        probabilities = {}
        
        for aspect, model in self.aspect_models.items():
            logger.info(f"Predicting probabilities for aspect: {aspect}")
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features)
                probabilities[aspect] = probs
            else:
                # Convert predictions to one-hot
                preds = model.predict(features)
                one_hot = np.zeros((len(preds), len(set(preds))))
                for i, pred in enumerate(preds):
                    one_hot[i, pred] = 1
                probabilities[aspect] = one_hot
        
        return probabilities
    
    def evaluate(self, features: Dict[str, Dict[str, Any]], aspect_labels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the model for each aspect.
        
        Args:
            features: Dictionary of features for each model
            aspect_labels: Dictionary of labels for each aspect
            
        Returns:
            Dictionary of evaluation metrics for each aspect
        """
        metrics = {}
        
        for aspect, model in self.aspect_models.items():
            if aspect not in aspect_labels:
                logger.warning(f"No labels found for aspect: {aspect}")
                continue
            
            logger.info(f"Evaluating for aspect: {aspect}")
            
            # Get predictions
            preds = model.predict(features)
            
            # Calculate metrics
            aspect_metrics = {
                'accuracy': accuracy_score(aspect_labels[aspect], preds),
                'precision': precision_score(aspect_labels[aspect], preds, average='weighted'),
                'recall': recall_score(aspect_labels[aspect], preds, average='weighted'),
                'f1': f1_score(aspect_labels[aspect], preds, average='weighted')
            }
            
            metrics[aspect] = aspect_metrics
            
            logger.info(f"Metrics for aspect {aspect}:")
            logger.info(f"Accuracy: {aspect_metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {aspect_metrics['f1']:.4f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the aspect-based sentiment model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save metadata
        metadata = {
            'aspects': self.aspects
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save individual models
        for aspect, model in self.aspect_models.items():
            model_path = os.path.join(os.path.dirname(filepath), f"aspect_{aspect}.pkl")
            
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        logger.info(f"Aspect-based sentiment model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, load_models: bool = True) -> 'AspectBasedSentimentModel':
        """
        Load an aspect-based sentiment model from disk.
        
        Args:
            filepath: Path to load the model from
            load_models: Whether to load the individual models
            
        Returns:
            Loaded aspect-based sentiment model
        """
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(aspects=metadata['aspects'])
        
        # Load individual models
        if load_models:
            model_dir = os.path.dirname(filepath)
            
            for aspect in metadata['aspects']:
                model_path = os.path.join(model_dir, f"aspect_{aspect}.pkl")
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    instance.aspect_models[aspect] = model
        
        logger.info(f"Aspect-based sentiment model loaded from {filepath}")
        return instance


# Example usage
if __name__ == "__main__":
    # Example data
    traditional_features = {
        'tfidf_features': np.random.rand(100, 1000),
        'lexical_features': np.random.rand(100, 10)
    }
    
    transformer_features = {
        'input_ids': np.random.randint(0, 1000, (100, config.MAX_SEQUENCE_LENGTH)),
        'attention_mask': np.random.randint(0, 2, (100, config.MAX_SEQUENCE_LENGTH))
    }
    
    features = {
        'traditional': traditional_features,
        'transformer': transformer_features
    }
    
    labels = np.random.randint(0, 3, 100)
    
    # Initialize models
    traditional_model = TraditionalModel(model_type="logistic_regression")
    transformer_model = TransformerModel(num_labels=3, num_epochs=1)
    
    # Initialize ensemble
    ensemble = EnsembleModel()
    ensemble.add_model('traditional', traditional_model, weight=0.3)
    ensemble.add_model('transformer', transformer_model, weight=0.7)
    
    # Fit ensemble
    ensemble.fit(features, labels)
    
    # Evaluate ensemble
    metrics = ensemble.evaluate(features, labels)
    
    print(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
    print(f"Ensemble F1 Score: {metrics['f1']:.4f}")

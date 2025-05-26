"""
Deep learning models for sentiment analysis using transformers.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys
import os
import logging
import pickle
from tqdm import tqdm

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """
    Dataset class for transformer models.
    """
    
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Args:
            input_ids: Token IDs from the tokenizer
            attention_mask: Attention mask from the tokenizer
            labels: Optional labels for the data
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Length of the dataset
        """
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with the item data
        """
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class TransformerModel:
    """
    Class for transformer-based models for sentiment analysis.
    """
    
    def __init__(self, model_name: str = config.TRANSFORMER_MODEL, 
                 num_labels: int = 3,
                 learning_rate: float = config.TRANSFORMER_LEARNING_RATE,
                 batch_size: int = config.TRANSFORMER_BATCH_SIZE,
                 num_epochs: int = config.TRANSFORMER_EPOCHS,
                 device: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the transformer model to use
            num_labels: Number of labels for classification
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Class mapping
        self.id2label = {i: str(i) for i in range(num_labels)}
        self.label2id = {str(i): i for i in range(num_labels)}
        
        # Initialize model
        self.model = self._create_model()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
    
    def _create_model(self):
        """
        Create a transformer model for sequence classification.
        
        Returns:
            The created model
        """
        # Load model configuration
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )
        
        return model
    
    def _create_dataloader(self, input_ids: np.ndarray, attention_mask: np.ndarray, 
                          labels: Optional[np.ndarray] = None, shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for the data.
        
        Args:
            input_ids: Token IDs from the tokenizer
            attention_mask: Attention mask from the tokenizer
            labels: Optional labels for the data
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the data
        """
        # Create dataset
        dataset = SentimentDataset(input_ids, attention_mask, labels)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        
        return dataloader
    
    def fit(self, features: Dict[str, np.ndarray], labels: np.ndarray, 
           val_features: Optional[Dict[str, np.ndarray]] = None, 
           val_labels: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Fit the model to the data.
        
        Args:
            features: Dictionary with 'input_ids' and 'attention_mask' features
            labels: Array of labels
            val_features: Optional validation features
            val_labels: Optional validation labels
            
        Returns:
            Dictionary with training history
        """
        # Verify features
        if 'input_ids' not in features or 'attention_mask' not in features:
            raise ValueError("Features must contain 'input_ids' and 'attention_mask'")
        
        # Create dataloaders
        train_dataloader = self._create_dataloader(
            features['input_ids'],
            features['attention_mask'],
            labels,
            shuffle=True
        )
        
        if val_features is not None and val_labels is not None:
            val_dataloader = self._create_dataloader(
                val_features['input_ids'],
                val_features['attention_mask'],
                val_labels,
                shuffle=False
            )
        else:
            val_dataloader = None
        
        # Create learning rate scheduler
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_true = []
            
            for batch in tqdm(train_dataloader, desc="Training"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true = batch['labels'].cpu().numpy()
                
                train_preds.extend(preds)
                train_true.extend(true)
            
            # Calculate metrics
            train_loss /= len(train_dataloader)
            train_accuracy = accuracy_score(train_true, train_preds)
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Validation
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Update metrics
                        val_loss += loss.item()
                        
                        # Get predictions
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        true = batch['labels'].cpu().numpy()
                        
                        val_preds.extend(preds)
                        val_true.extend(true)
                
                # Calculate metrics
                val_loss /= len(val_dataloader)
                val_accuracy = accuracy_score(val_true, val_preds)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return history
    
    def predict(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict labels for the data.
        
        Args:
            features: Dictionary with 'input_ids' and 'attention_mask' features
            
        Returns:
            Array of predicted labels
        """
        # Verify features
        if 'input_ids' not in features or 'attention_mask' not in features:
            raise ValueError("Features must contain 'input_ids' and 'attention_mask'")
        
        # Create dataloader
        dataloader = self._create_dataloader(
            features['input_ids'],
            features['attention_mask'],
            shuffle=False
        )
        
        # Prediction loop
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for the data.
        
        Args:
            features: Dictionary with 'input_ids' and 'attention_mask' features
            
        Returns:
            Array of predicted class probabilities
        """
        # Verify features
        if 'input_ids' not in features or 'attention_mask' not in features:
            raise ValueError("Features must contain 'input_ids' and 'attention_mask'")
        
        # Create dataloader
        dataloader = self._create_dataloader(
            features['input_ids'],
            features['attention_mask'],
            shuffle=False
        )
        
        # Prediction loop
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get probabilities
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def evaluate(self, features: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on the data.
        
        Args:
            features: Dictionary with 'input_ids' and 'attention_mask' features
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
        
        logger.info(f"Evaluation metrics for transformer model:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'TransformerModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to load the model from
            device: Device to use for the model
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model
        instance = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels'],
            device=device
        )
        
        # Load state dict
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Set class mapping
        instance.id2label = checkpoint['id2label']
        instance.label2id = checkpoint['label2id']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


# Example usage
if __name__ == "__main__":
    # Example data
    input_ids = np.random.randint(0, 1000, (100, config.MAX_SEQUENCE_LENGTH))
    attention_mask = np.random.randint(0, 2, (100, config.MAX_SEQUENCE_LENGTH))
    labels = np.random.randint(0, 3, 100)
    
    features = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Initialize model
    model = TransformerModel(num_labels=3, num_epochs=1)
    
    # Fit model
    history = model.fit(features, labels)
    
    # Evaluate model
    metrics = model.evaluate(features, labels)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

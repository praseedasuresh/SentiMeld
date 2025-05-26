"""
Main entry point for the SentiMeld sentiment analysis tool.
"""
import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import project modules
import config
from data.data_loader import DataLoader
from preprocessing.cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.traditional import TraditionalModel
from models.deep_learning import TransformerModel
from models.ensemble import EnsembleModel, AspectBasedSentimentModel
from visualization.plots import plot_confusion_matrix, plot_feature_importance, plot_training_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SentiMeld: Advanced Sentiment Analysis Tool')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'analyze', 'evaluate'],
                       help='Mode to run the tool in')
    
    # Data options
    parser.add_argument('--dataset', type=str, default='imdb',
                       help='Dataset to use (imdb, twitter, or custom)')
    parser.add_argument('--data_path', type=str,
                       help='Path to custom dataset')
    
    # Model options
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['traditional', 'transformer', 'ensemble'],
                       help='Model type to use')
    parser.add_argument('--traditional_model', type=str, default='logistic_regression',
                       choices=['naive_bayes', 'svm', 'logistic_regression', 'random_forest', 'gradient_boosting'],
                       help='Traditional model type to use')
    parser.add_argument('--transformer_model', type=str, default=config.TRANSFORMER_MODEL,
                       help='Transformer model to use')
    
    # Feature options
    parser.add_argument('--feature_method', type=str, default='ensemble',
                       choices=['bow', 'tfidf', 'transformer', 'ensemble'],
                       help='Feature extraction method')
    parser.add_argument('--use_topic_features', action='store_true',
                       help='Whether to use topic modeling features')
    parser.add_argument('--use_lexical_features', action='store_true',
                       help='Whether to use lexical features')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=config.TRANSFORMER_EPOCHS,
                       help='Number of epochs for transformer training')
    parser.add_argument('--batch_size', type=int, default=config.TRANSFORMER_BATCH_SIZE,
                       help='Batch size for transformer training')
    parser.add_argument('--learning_rate', type=float, default=config.TRANSFORMER_LEARNING_RATE,
                       help='Learning rate for transformer training')
    
    # Aspect-based options
    parser.add_argument('--aspect_based', action='store_true',
                       help='Whether to use aspect-based sentiment analysis')
    parser.add_argument('--domain', type=str, default='product_reviews',
                       choices=['product_reviews', 'tweets'],
                       help='Domain for aspect-based sentiment analysis')
    
    # Analysis options
    parser.add_argument('--text', type=str,
                       help='Text to analyze (for analyze mode)')
    parser.add_argument('--text_file', type=str,
                       help='Path to file containing text to analyze (for analyze mode)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default=str(config.MODEL_ARTIFACTS_DIR),
                       help='Directory to save model artifacts')
    parser.add_argument('--model_name', type=str, default='sentimeld_model',
                       help='Name of the model to save/load')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Whether to generate visualizations')
    
    return parser.parse_args()

def train(args):
    """Train a sentiment analysis model."""
    logger.info(f"Training a {args.model} model on {args.dataset} dataset")
    
    # Load data
    data_loader = DataLoader(
        dataset_name=args.dataset,
        custom_path=args.data_path
    )
    data = data_loader.process_and_prepare_data()
    
    # Get data splits
    train_data = data['data_splits']['train']
    val_data = data['data_splits']['val']
    test_data = data['data_splits']['test']
    
    # Initialize text cleaner
    cleaner = TextCleaner()
    
    # Clean text
    logger.info("Cleaning text data...")
    train_texts = cleaner.clean_texts(train_data['text'].tolist())
    val_texts = cleaner.clean_texts(val_data['text'].tolist())
    test_texts = cleaner.clean_texts(test_data['text'].tolist())
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        method=args.feature_method,
        transformer_model=args.transformer_model,
        use_topic_features=args.use_topic_features,
        use_lexical_features=args.use_lexical_features
    )
    
    # Extract features
    logger.info("Extracting features...")
    train_features = feature_extractor.extract_features(train_texts, fit=True)
    val_features = feature_extractor.extract_features(val_texts, fit=False)
    test_features = feature_extractor.extract_features(test_texts, fit=False)
    
    # Get labels
    train_labels = train_data['label'].values
    val_labels = val_data['label'].values
    test_labels = test_data['label'].values
    
    # Train model
    if args.model == 'traditional':
        # Train traditional model
        model = TraditionalModel(model_type=args.traditional_model)
        model.fit(train_features, train_labels)
        
        # Evaluate model
        metrics = model.evaluate(test_features, test_labels)
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model_name}_traditional.pkl")
        model.save(model_path)
        
        # Save feature extractor
        feature_extractor_path = os.path.join(args.output_dir, f"{args.model_name}_feature_extractor.pkl")
        feature_extractor.save(feature_extractor_path)
        
        # Visualize
        if args.visualize:
            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                classes=model.classes_,
                output_path=os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix.png")
            )
            
            # Plot feature importance
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                feature_names = feature_extractor.get_feature_names()
                flat_feature_names = []
                for key, names in feature_names.items():
                    flat_feature_names.extend(names)
                
                plot_feature_importance(
                    model.get_feature_importances(flat_feature_names),
                    output_path=os.path.join(args.output_dir, f"{args.model_name}_feature_importance.png")
                )
    
    elif args.model == 'transformer':
        # Train transformer model
        model = TransformerModel(
            model_name=args.transformer_model,
            num_labels=len(set(train_labels)),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        
        # Fit model
        history = model.fit(
            train_features,
            train_labels,
            val_features=val_features,
            val_labels=val_labels
        )
        
        # Evaluate model
        metrics = model.evaluate(test_features, test_labels)
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model_name}_transformer.pt")
        model.save(model_path)
        
        # Save feature extractor
        feature_extractor_path = os.path.join(args.output_dir, f"{args.model_name}_feature_extractor.pkl")
        feature_extractor.save(feature_extractor_path)
        
        # Visualize
        if args.visualize:
            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                classes=list(range(len(set(train_labels)))),
                output_path=os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix.png")
            )
            
            # Plot training history
            plot_training_history(
                history,
                output_path=os.path.join(args.output_dir, f"{args.model_name}_training_history.png")
            )
    
    elif args.model == 'ensemble':
        # Train ensemble model
        ensemble = EnsembleModel()
        
        # Add traditional model
        traditional_model = TraditionalModel(model_type=args.traditional_model)
        ensemble.add_model('traditional', traditional_model, weight=config.ENSEMBLE_WEIGHTS['traditional'])
        
        # Add transformer model
        transformer_model = TransformerModel(
            model_name=args.transformer_model,
            num_labels=len(set(train_labels)),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        ensemble.add_model('transformer', transformer_model, weight=config.ENSEMBLE_WEIGHTS['transformer'])
        
        # Prepare features for each model
        ensemble_train_features = {
            'traditional': train_features,
            'transformer': train_features
        }
        
        ensemble_val_features = {
            'traditional': val_features,
            'transformer': val_features
        }
        
        ensemble_test_features = {
            'traditional': test_features,
            'transformer': test_features
        }
        
        # Fit ensemble
        history = ensemble.fit(
            ensemble_train_features,
            train_labels,
            val_features=ensemble_val_features,
            val_labels=val_labels
        )
        
        # Evaluate ensemble
        metrics = ensemble.evaluate(ensemble_test_features, test_labels)
        
        # Save ensemble
        os.makedirs(args.output_dir, exist_ok=True)
        ensemble_path = os.path.join(args.output_dir, f"{args.model_name}_ensemble.pkl")
        ensemble.save(ensemble_path)
        
        # Save feature extractor
        feature_extractor_path = os.path.join(args.output_dir, f"{args.model_name}_feature_extractor.pkl")
        feature_extractor.save(feature_extractor_path)
        
        # Visualize
        if args.visualize:
            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['ensemble']['confusion_matrix'],
                classes=list(range(len(set(train_labels)))),
                output_path=os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix.png")
            )
    
    # Train aspect-based model if requested
    if args.aspect_based:
        logger.info("Training aspect-based sentiment model...")
        
        # Get aspects for the domain
        aspects = config.ASPECTS.get(args.domain, [])
        
        if not aspects:
            logger.warning(f"No aspects found for domain: {args.domain}")
            return
        
        # Create aspect labels (simulated for this example)
        # In a real implementation, you would have actual aspect labels
        aspect_labels = {}
        for aspect in aspects:
            # Simulate aspect labels (random for this example)
            aspect_labels[aspect] = np.random.randint(0, 3, len(train_labels))
        
        # Create aspect-based model
        aspect_model = AspectBasedSentimentModel(aspects=aspects)
        
        # Fit aspect-based model
        aspect_model.fit(train_features, aspect_labels)
        
        # Save aspect-based model
        aspect_model_path = os.path.join(args.output_dir, f"{args.model_name}_aspect_based.pkl")
        aspect_model.save(aspect_model_path)
    
    logger.info("Training completed successfully!")
    
    # Print metrics
    if args.model == 'ensemble':
        logger.info(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.4f}")
        logger.info(f"Ensemble F1 Score: {metrics['ensemble']['f1']:.4f}")
    else:
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")

def analyze(args):
    """Analyze sentiment of text."""
    # Get text to analyze
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        logger.error("No text provided for analysis")
        return
    
    # Load feature extractor
    feature_extractor_path = os.path.join(args.output_dir, f"{args.model_name}_feature_extractor.pkl")
    if not os.path.exists(feature_extractor_path):
        logger.error(f"Feature extractor not found at {feature_extractor_path}")
        return
    
    feature_extractor = FeatureExtractor.load(feature_extractor_path)
    
    # Initialize text cleaner
    cleaner = TextCleaner()
    
    # Clean text
    cleaned_texts = cleaner.clean_texts(texts)
    
    # Extract features
    features = feature_extractor.extract_features(cleaned_texts, fit=False)
    
    # Load model
    if args.model == 'traditional':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_traditional.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = TraditionalModel.load(model_path)
        
        # Predict
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
    elif args.model == 'transformer':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_transformer.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = TransformerModel.load(model_path)
        
        # Predict
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
    elif args.model == 'ensemble':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_ensemble.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = EnsembleModel.load(model_path)
        
        # Prepare features for each model
        ensemble_features = {}
        for name in model.models:
            ensemble_features[name] = features
        
        # Predict
        predictions = model.predict(ensemble_features)
        probabilities = model.predict_proba(ensemble_features)
    
    # Map predictions to sentiment labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiments = [sentiment_map.get(pred, str(pred)) for pred in predictions]
    
    # Print results
    logger.info("Sentiment Analysis Results:")
    for i, (text, sentiment, probs) in enumerate(zip(texts, sentiments, probabilities)):
        logger.info(f"Text {i+1}: {text.strip()}")
        logger.info(f"Sentiment: {sentiment}")
        logger.info(f"Confidence: {np.max(probs):.4f}")
        logger.info("---")
    
    # Analyze aspects if requested
    if args.aspect_based:
        aspect_model_path = os.path.join(args.output_dir, f"{args.model_name}_aspect_based.pkl")
        if not os.path.exists(aspect_model_path):
            logger.warning(f"Aspect-based model not found at {aspect_model_path}")
            return
        
        aspect_model = AspectBasedSentimentModel.load(aspect_model_path)
        
        # Predict aspects
        aspect_predictions = aspect_model.predict(features)
        
        # Print aspect results
        logger.info("Aspect-Based Sentiment Analysis Results:")
        for i, text in enumerate(texts):
            logger.info(f"Text {i+1}: {text.strip()}")
            for aspect, preds in aspect_predictions.items():
                aspect_sentiment = sentiment_map.get(preds[i], str(preds[i]))
                logger.info(f"Aspect '{aspect}': {aspect_sentiment}")
            logger.info("---")

def evaluate(args):
    """Evaluate a trained model on a test dataset."""
    logger.info(f"Evaluating {args.model} model on {args.dataset} dataset")
    
    # Load data
    data_loader = DataLoader(
        dataset_name=args.dataset,
        custom_path=args.data_path
    )
    data = data_loader.process_and_prepare_data()
    
    # Get test data
    test_data = data['data_splits']['test']
    
    # Initialize text cleaner
    cleaner = TextCleaner()
    
    # Clean text
    test_texts = cleaner.clean_texts(test_data['text'].tolist())
    
    # Load feature extractor
    feature_extractor_path = os.path.join(args.output_dir, f"{args.model_name}_feature_extractor.pkl")
    if not os.path.exists(feature_extractor_path):
        logger.error(f"Feature extractor not found at {feature_extractor_path}")
        return
    
    feature_extractor = FeatureExtractor.load(feature_extractor_path)
    
    # Extract features
    test_features = feature_extractor.extract_features(test_texts, fit=False)
    
    # Get labels
    test_labels = test_data['label'].values
    
    # Load and evaluate model
    if args.model == 'traditional':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_traditional.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = TraditionalModel.load(model_path)
        metrics = model.evaluate(test_features, test_labels)
        
    elif args.model == 'transformer':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_transformer.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = TransformerModel.load(model_path)
        metrics = model.evaluate(test_features, test_labels)
        
    elif args.model == 'ensemble':
        model_path = os.path.join(args.output_dir, f"{args.model_name}_ensemble.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = EnsembleModel.load(model_path)
        
        # Prepare features for each model
        ensemble_features = {}
        for name in model.models:
            ensemble_features[name] = test_features
        
        metrics = model.evaluate(ensemble_features, test_labels)
    
    # Print metrics
    if args.model == 'ensemble':
        logger.info(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.4f}")
        logger.info(f"Ensemble Precision: {metrics['ensemble']['precision']:.4f}")
        logger.info(f"Ensemble Recall: {metrics['ensemble']['recall']:.4f}")
        logger.info(f"Ensemble F1 Score: {metrics['ensemble']['f1']:.4f}")
        
        # Print individual model metrics
        for name, model_metrics in metrics['models'].items():
            logger.info(f"{name.capitalize()} Model Accuracy: {model_metrics['accuracy']:.4f}")
            logger.info(f"{name.capitalize()} Model F1 Score: {model_metrics['f1']:.4f}")
    else:
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Visualize
    if args.visualize:
        # Plot confusion matrix
        if args.model == 'ensemble':
            confusion_matrix = metrics['ensemble']['confusion_matrix']
            classes = list(range(len(set(test_labels))))
        else:
            confusion_matrix = metrics['confusion_matrix']
            classes = model.classes_ if hasattr(model, 'classes_') else list(range(len(set(test_labels))))
        
        plot_confusion_matrix(
            confusion_matrix,
            classes=classes,
            output_path=os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix_eval.png")
        )

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'analyze':
        analyze(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()

"""
Visualization utilities for the SentiMeld project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union, Optional, Tuple
import os
import sys
import logging
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(confusion_matrix: List[List[int]], 
                         classes: List[Any],
                         output_path: Optional[str] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         cmap: str = 'Blues') -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix to plot
        classes: List of class labels
        output_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
        cmap: Colormap to use
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy array
    cm = np.array(confusion_matrix)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
               xticklabels=classes, yticklabels=classes)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
    
    return plt.gcf()

def plot_feature_importance(feature_importances: Dict[str, float],
                           top_n: int = 20,
                           output_path: Optional[str] = None,
                           title: str = 'Feature Importance') -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        feature_importances: Dictionary mapping feature names to importances
        top_n: Number of top features to plot
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N features
    top_features = sorted_features[:top_n]
    
    # Create dataframe
    df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot feature importances
    sns.barplot(x='Importance', y='Feature', data=df)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    return plt.gcf()

def plot_training_history(history: Dict[str, List[float]],
                         output_path: Optional[str] = None,
                         title: str = 'Training History') -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation')
    
    axes[0].set_title('Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    
    # Plot accuracy
    axes[1].plot(history['train_accuracy'], label='Train')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation')
    
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {output_path}")
    
    return fig

def plot_word_cloud(texts: List[str],
                   output_path: Optional[str] = None,
                   title: str = 'Word Cloud',
                   max_words: int = 200,
                   background_color: str = 'white') -> plt.Figure:
    """
    Plot a word cloud from texts.
    
    Args:
        texts: List of texts to generate word cloud from
        output_path: Path to save the plot
        title: Title of the plot
        max_words: Maximum number of words to include
        background_color: Background color of the word cloud
        
    Returns:
        Matplotlib figure
    """
    # Combine texts
    text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        max_words=max_words,
        background_color=background_color,
        width=800,
        height=400
    ).generate(text)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    plt.title(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Word cloud saved to {output_path}")
    
    return plt.gcf()

def plot_sentiment_distribution(sentiments: List[str],
                              output_path: Optional[str] = None,
                              title: str = 'Sentiment Distribution') -> plt.Figure:
    """
    Plot sentiment distribution.
    
    Args:
        sentiments: List of sentiment labels
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Count sentiments
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot sentiment distribution
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sentiment distribution plot saved to {output_path}")
    
    return plt.gcf()

def plot_aspect_based_sentiment(aspect_sentiments: Dict[str, List[str]],
                              output_path: Optional[str] = None,
                              title: str = 'Aspect-Based Sentiment Analysis') -> plt.Figure:
    """
    Plot aspect-based sentiment analysis results.
    
    Args:
        aspect_sentiments: Dictionary mapping aspects to sentiment labels
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Create dataframe
    data = []
    for aspect, sentiments in aspect_sentiments.items():
        # Count sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Add to data
        for sentiment, count in sentiment_counts.items():
            data.append({
                'Aspect': aspect,
                'Sentiment': sentiment,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot aspect-based sentiment
    sns.barplot(x='Aspect', y='Count', hue='Sentiment', data=df)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Aspect', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Rotate x-axis labels if there are many aspects
    if len(aspect_sentiments) > 5:
        plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Aspect-based sentiment plot saved to {output_path}")
    
    return plt.gcf()

def create_interactive_dashboard(texts: List[str],
                               sentiments: List[str],
                               probabilities: List[List[float]],
                               aspect_sentiments: Optional[Dict[str, List[str]]] = None,
                               output_path: Optional[str] = None,
                               title: str = 'Sentiment Analysis Dashboard') -> go.Figure:
    """
    Create an interactive dashboard for sentiment analysis results.
    
    Args:
        texts: List of texts
        sentiments: List of sentiment labels
        probabilities: List of probability arrays
        aspect_sentiments: Optional dictionary mapping aspects to sentiment labels
        output_path: Path to save the dashboard
        title: Title of the dashboard
        
    Returns:
        Plotly figure
    """
    # Create dataframe
    df = pd.DataFrame({
        'Text': texts,
        'Sentiment': sentiments
    })
    
    # Add probabilities
    for i, prob in enumerate(probabilities[0]):
        df[f'Probability_{i}'] = [p[i] for p in probabilities]
    
    # Create subplots
    n_rows = 2 if aspect_sentiments else 1
    fig = make_subplots(
        rows=n_rows, 
        cols=2,
        subplot_titles=('Sentiment Distribution', 'Sentiment Confidence', 
                       'Aspect-Based Sentiment' if aspect_sentiments else None),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
              [{'type': 'bar', 'colspan': 2}, None]] if aspect_sentiments else
              [[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Plot sentiment distribution
    sentiment_counts = df['Sentiment'].value_counts()
    fig.add_trace(
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            name='Sentiment Distribution'
        ),
        row=1, col=1
    )
    
    # Plot sentiment confidence
    sentiment_confidence = df.groupby('Sentiment')[['Probability_0', 'Probability_1', 'Probability_2']].mean()
    fig.add_trace(
        go.Bar(
            x=sentiment_confidence.index,
            y=sentiment_confidence.mean(axis=1),
            name='Confidence'
        ),
        row=1, col=2
    )
    
    # Plot aspect-based sentiment if provided
    if aspect_sentiments:
        # Create dataframe
        aspect_data = []
        for aspect, aspect_sentiments_list in aspect_sentiments.items():
            # Count sentiments
            sentiment_counts = pd.Series(aspect_sentiments_list).value_counts()
            
            # Add to data
            for sentiment, count in sentiment_counts.items():
                aspect_data.append({
                    'Aspect': aspect,
                    'Sentiment': sentiment,
                    'Count': count
                })
        
        aspect_df = pd.DataFrame(aspect_data)
        
        # Plot aspect-based sentiment
        for sentiment in aspect_df['Sentiment'].unique():
            sentiment_data = aspect_df[aspect_df['Sentiment'] == sentiment]
            fig.add_trace(
                go.Bar(
                    x=sentiment_data['Aspect'],
                    y=sentiment_data['Count'],
                    name=sentiment
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600 if aspect_sentiments else 400,
        width=1000,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save if output path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Interactive dashboard saved to {output_path}")
    
    return fig

def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]],
                         metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                         output_path: Optional[str] = None,
                         title: str = 'Model Comparison') -> plt.Figure:
    """
    Plot model comparison.
    
    Args:
        model_metrics: Dictionary mapping model names to metrics
        metrics: List of metrics to plot
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Create dataframe
    data = []
    for model, model_metric in model_metrics.items():
        for metric in metrics:
            if metric in model_metric:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': model_metric[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot model comparison
    sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {output_path}")
    
    return plt.gcf()

def plot_emotion_distribution(emotions: List[str],
                            output_path: Optional[str] = None,
                            title: str = 'Emotion Distribution') -> plt.Figure:
    """
    Plot emotion distribution.
    
    Args:
        emotions: List of emotion labels
        output_path: Path to save the plot
        title: Title of the plot
        
    Returns:
        Matplotlib figure
    """
    # Count emotions
    emotion_counts = pd.Series(emotions).value_counts()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot emotion distribution
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Rotate x-axis labels if there are many emotions
    if len(emotion_counts) > 5:
        plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Emotion distribution plot saved to {output_path}")
    
    return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Example data
    confusion_matrix = [
        [10, 2, 1],
        [1, 15, 3],
        [0, 2, 20]
    ]
    
    classes = ['negative', 'neutral', 'positive']
    
    # Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        classes=classes,
        output_path='confusion_matrix.png'
    )
    
    # Example feature importances
    feature_importances = {
        'good': 0.1,
        'bad': 0.08,
        'excellent': 0.07,
        'terrible': 0.06,
        'amazing': 0.05,
        'awful': 0.04,
        'great': 0.03,
        'horrible': 0.02,
        'fantastic': 0.01,
        'poor': 0.009
    }
    
    # Plot feature importance
    plot_feature_importance(
        feature_importances,
        output_path='feature_importance.png'
    )
    
    # Example training history
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25],
        'train_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.8, 0.85, 0.87]
    }
    
    # Plot training history
    plot_training_history(
        history,
        output_path='training_history.png'
    )

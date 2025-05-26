"""
Interactive dashboard for the SentiMeld sentiment analysis tool.
This module provides a web-based dashboard for exploring sentiment analysis results.
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
import json
from collections import Counter

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.data_loader import DataLoader
from preprocessing.cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.traditional import TraditionalModel
from models.ensemble import EnsembleModel
from visualization.plots import (
    create_interactive_dashboard, 
    plot_sentiment_distribution,
    plot_aspect_based_sentiment,
    plot_word_cloud
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentDashboard:
    """
    Interactive dashboard for sentiment analysis results.
    """
    
    def __init__(self, model_name: str = "sentimeld_model"):
        """
        Initialize the dashboard.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.app = dash.Dash(__name__, title="SentiMeld Dashboard")
        self.model = None
        self.feature_extractor = None
        self.results = []
        
        # Load model and feature extractor
        self._load_model()
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _load_model(self):
        """Load the model and feature extractor."""
        logger.info("Loading model and feature extractor...")
        
        try:
            # Load feature extractor
            feature_extractor_path = os.path.join(
                config.MODEL_ARTIFACTS_DIR, 
                f"{self.model_name}_feature_extractor.pkl"
            )
            self.feature_extractor = FeatureExtractor.load(feature_extractor_path)
            
            # Load ensemble model
            model_path = os.path.join(
                config.MODEL_ARTIFACTS_DIR, 
                f"{self.model_name}_ensemble.pkl"
            )
            self.model = EnsembleModel.load(model_path)
            
            logger.info("Model and feature extractor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to loading just the traditional model
            try:
                logger.info("Attempting to load traditional model as fallback...")
                model_path = os.path.join(
                    config.MODEL_ARTIFACTS_DIR, 
                    f"{self.model_name}_traditional.pkl"
                )
                self.model = TraditionalModel.load(model_path)
                logger.info("Traditional model loaded successfully")
            except Exception as nested_e:
                logger.error(f"Error loading traditional model: {nested_e}")
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("SentiMeld Dashboard", style={'textAlign': 'center'}),
                html.P("Interactive dashboard for sentiment analysis results", style={'textAlign': 'center'})
            ], style={'margin': '20px 0'}),
            
            # Input section
            html.Div([
                html.H3("Analyze Text"),
                dcc.Textarea(
                    id='text-input',
                    placeholder='Enter text to analyze...',
                    style={'width': '100%', 'height': 150}
                ),
                html.Button('Analyze', id='analyze-button', n_clicks=0, 
                           style={'margin': '10px 0', 'padding': '10px 20px'}),
                html.Div(id='analysis-output')
            ], style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            # Results section
            html.Div([
                html.H3("Analysis Results"),
                html.Div(id='sentiment-gauge', style={'margin': '20px 0'}),
                html.Div([
                    html.Div([
                        html.H4("Sentiment Probabilities"),
                        dcc.Graph(id='sentiment-probs')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("Word Cloud"),
                        dcc.Graph(id='word-cloud')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
            ], style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            # History section
            html.Div([
                html.H3("Analysis History"),
                html.Button('Clear History', id='clear-history', n_clicks=0,
                           style={'margin': '10px 0', 'padding': '5px 10px'}),
                html.Div(id='history-table')
            ], style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            # Store for results
            dcc.Store(id='results-store', data=[])
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [Output('analysis-output', 'children'),
             Output('sentiment-gauge', 'children'),
             Output('sentiment-probs', 'figure'),
             Output('word-cloud', 'figure'),
             Output('results-store', 'data')],
            [Input('analyze-button', 'n_clicks')],
            [State('text-input', 'value'),
             State('results-store', 'data')],
            prevent_initial_call=True
        )
        def analyze_text(n_clicks, text, results):
            if not text:
                return "Please enter some text to analyze.", None, {}, {}, results
            
            # Clean the text
            cleaner = TextCleaner()
            cleaned_text = cleaner.clean_texts([text])[0]
            
            # Extract features
            features = self.feature_extractor.extract_features([cleaned_text], fit=False)
            
            # Make prediction
            if isinstance(self.model, EnsembleModel):
                # For ensemble model, we need to structure the features differently
                model_features = {'traditional': features}
                sentiment = self.model.predict(model_features)[0]
                probs = self.model.predict_proba(model_features)[0]
            else:
                sentiment = self.model.predict(features)[0]
                probs = self.model.predict_proba(features)[0]
            
            # Get sentiment label
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
                sentiment_label = 'positive' if sentiment == 1 else 'negative'
            else:
                classes = [0, 1]  # Default to binary classification
                sentiment_label = 'positive' if sentiment == 1 else 'negative'
            
            # Create gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probs[1] * 100,  # Assuming binary classification with positive=1
                title={'text': f"Sentiment: {sentiment_label.upper()}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            gauge_fig.update_layout(height=300)
            
            # Create probability bar chart
            prob_labels = ['Negative', 'Positive'] if len(probs) == 2 else [f'Class {i}' for i in range(len(probs))]
            probs_fig = go.Figure(data=[
                go.Bar(
                    x=prob_labels,
                    y=[p * 100 for p in probs],
                    marker_color=['lightcoral', 'lightgreen'] if len(probs) == 2 else None
                )
            ])
            
            probs_fig.update_layout(
                title="Sentiment Probabilities (%)",
                yaxis_title="Probability (%)",
                height=300
            )
            
            # Create word cloud
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            # Generate word cloud
            wordcloud = WordCloud(
                background_color='white',
                max_words=100,
                width=800,
                height=400
            ).generate(cleaned_text)
            
            # Convert to image
            img = BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            
            # Create plotly figure with the image
            wc_fig = go.Figure()
            
            wc_fig.add_layout_image(
                dict(
                    source=f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}',
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    sizing="stretch",
                    layer="below"
                )
            )
            
            wc_fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            # Add to results
            result = {
                'text': text,
                'sentiment': sentiment_label,
                'probabilities': probs.tolist(),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
            
            return (
                f"Analysis complete! Sentiment: {sentiment_label.upper()}",
                dcc.Graph(figure=gauge_fig),
                probs_fig,
                wc_fig,
                results
            )
        
        @self.app.callback(
            Output('history-table', 'children'),
            [Input('results-store', 'data'),
             Input('clear-history', 'n_clicks')]
        )
        def update_history(results, clear_clicks):
            # Clear history if button clicked
            if clear_clicks and clear_clicks > 0:
                return html.P("No analysis history.")
            
            if not results:
                return html.P("No analysis history.")
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Time"),
                    html.Th("Text"),
                    html.Th("Sentiment"),
                    html.Th("Confidence")
                ]))
            ]
            
            rows = []
            for result in results:
                # Get confidence
                probs = result['probabilities']
                if result['sentiment'] == 'positive':
                    confidence = probs[1] * 100
                else:
                    confidence = probs[0] * 100
                
                # Create row
                row = html.Tr([
                    html.Td(result['timestamp']),
                    html.Td(result['text'][:50] + '...' if len(result['text']) > 50 else result['text']),
                    html.Td(result['sentiment'].upper()),
                    html.Td(f"{confidence:.1f}%")
                ])
                
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            
            return html.Table(table_header + table_body, style={'width': '100%', 'border': '1px solid #ddd'})
    
    def run(self, host: str = config.DASHBOARD_HOST, port: int = config.DASHBOARD_PORT, debug: bool = config.DASHBOARD_DEBUG):
        """
        Run the dashboard.
        
        Args:
            host: Host to run the dashboard on
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        self.app.run_server(host=host, port=port, debug=debug)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SentiMeld Dashboard')
    
    parser.add_argument('--model', type=str, default='sentimeld_model',
                       help='Name of the model to use')
    parser.add_argument('--host', type=str, default=config.DASHBOARD_HOST,
                       help='Host to run the dashboard on')
    parser.add_argument('--port', type=int, default=config.DASHBOARD_PORT,
                       help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                       help='Whether to run in debug mode')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting SentiMeld Dashboard with model {args.model}")
    
    dashboard = SentimentDashboard(model_name=args.model)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

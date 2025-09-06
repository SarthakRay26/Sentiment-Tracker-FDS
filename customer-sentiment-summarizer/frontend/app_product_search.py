#!/usr/bin/env python3
"""
Enhanced Customer Sentiment Summarizer with Product Search
"""

import gradio as gr
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time

# Add the parent directory to sys.path to import from backend
sys.path.append(str(Path(__file__).parent.parent))

from backend.main_simple import CustomerSentimentSummarizer

class SentimentApp:
    def __init__(self):
        """Initialize the Sentiment Analysis App"""
        print("🚀 Starting Customer Sentiment Summarizer Chatbot...")
        print("🚀 Initializing AI models...")
        
        # Initialize the main summarizer
        self.summarizer = CustomerSentimentSummarizer()
        
        print("✅ AI models loaded successfully!")
    
    def analyze_product_reviews(self, product_query: str, sample_size: int) -> Tuple[str, str, Optional[gr.Plot]]:
        """Analyze product reviews by searching for relevant datasets"""
        if not product_query.strip():
            error_msg = "❌ Please enter a product name or category to search for reviews."
            return error_msg, error_msg, None
        
        try:
            print(f"🔍 Searching for reviews related to: {product_query}")
            
            # Map product queries to appropriate datasets
            dataset_mapping = self._map_product_to_dataset(product_query.lower())
            
            if not dataset_mapping:
                error_msg = f"❌ No suitable dataset found for '{product_query}'. Try 'Amazon products' for general analysis."
                return error_msg, error_msg, None
            
            dataset_id = dataset_mapping['dataset_id']
            
            print(f"📦 Using dataset: {dataset_id}")
            
            # Import the Kaggle importer
            sys.path.append(str(Path(__file__).parent.parent / 'backend'))
            from kaggle_importer import KaggleDatasetImporter
            
            # Initialize importer
            importer = KaggleDatasetImporter()
            
            # Load the dataset with product filtering
            if dataset_id == 'datafiniti/consumer-reviews-of-amazon-products':
                # Use the Amazon-specific processor with product filtering
                reviews, ratings = importer.process_amazon_product_reviews(
                    dataset_id=dataset_id,
                    product_filter=product_query
                )
            elif dataset_id == 'snap/amazon-fine-food-reviews':
                # Use general loader for food reviews
                dataset_path = importer.download_dataset(dataset_id)
                reviews, ratings = importer.load_dataset_for_analysis(
                    dataset_path,
                    text_column='Text',
                    rating_column='Score'
                )
            else:
                # Fallback to general loader
                dataset_path = importer.download_dataset(dataset_id)
                reviews, ratings = importer.load_dataset_for_analysis(
                    dataset_path,
                    text_column='reviews.text',
                    rating_column='reviews.rating'
                )
            
            if not reviews:
                error_msg = f"❌ No reviews found for '{product_query}'. Please try a different search term."
                return error_msg, error_msg, None
            
            # Sample the data if too large
            if len(reviews) > sample_size:
                import random
                indices = random.sample(range(len(reviews)), sample_size)
                reviews = [reviews[i] for i in indices]
                if ratings:
                    ratings = [ratings[i] for i in indices if i < len(ratings)]
                print(f"📊 Sampled {sample_size} reviews from {len(reviews)} total")
            
            print(f"🔍 Analyzing {len(reviews)} reviews for '{product_query}'...")
            
            # Analyze the reviews
            result = self.summarizer.analyze_customer_feedback(reviews)
            
            # Format results with product info
            product_info = f"🔍 **Product Review Analysis**\n\n"
            product_info += f"**Search Query:** {product_query}\n"
            product_info += f"**Reviews Analyzed:** {len(reviews)}\n"
            product_info += f"**Data Source:** {dataset_mapping['description']}\n"
            if ratings:
                valid_ratings = [r for r in ratings if r is not None and r > 0]
                if valid_ratings:
                    avg_rating = sum(valid_ratings) / len(valid_ratings)
                    product_info += f"**Average Rating:** {avg_rating:.1f}/5.0\n"
            product_info += "\n"
            
            if not result or 'error' in result:
                error_msg = "❌ Analysis failed. Please try a different search term."
                return error_msg, error_msg, None
            
            # Extract data from result
            sentiment_data = result.get('sentiment', {})
            summary_data = result.get('summary', {})
            
            # Debug: Print what we got
            print(f"🔍 Result keys: {list(result.keys())}")
            print(f"📊 Sentiment data: {sentiment_data}")
            print(f"📝 Summary data: {summary_data}")
            
            # Format outputs
            sentiment_result = product_info + self._format_sentiment_output_multiple(sentiment_data, len(reviews))
            summary_result = self._format_summary_output(summary_data)
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(sentiment_data)
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, None
    
    def _map_product_to_dataset(self, product_query: str) -> Optional[Dict]:
        """Map product search queries to appropriate datasets with precise matching"""
        
        product_query_lower = product_query.lower().strip()
        
        # Define dataset mappings with precise word matching
        datasets = [
            {
                'dataset_id': 'datafiniti/consumer-reviews-of-amazon-products',
                'description': 'Amazon Product Reviews (28K+ reviews)',
                'keywords': ['iphone', 'samsung', 'galaxy', 'phone', 'smartphone', 'laptop', 'macbook', 'computer', 'tablet', 'ipad', 'headphones', 'speaker', 'camera', 'tv', 'television', 'appliance', 'electronics', 'device', 'gadget'],
                'exact_matches': ['iphone 15', 'iphone 14', 'samsung galaxy', 'macbook pro', 'macbook air', 'ipad pro']
            },
            {
                'dataset_id': 'snap/amazon-fine-food-reviews',
                'description': 'Amazon Fine Food Reviews',
                'keywords': ['food', 'snack', 'coffee', 'tea', 'drink', 'beverage', 'nutrition', 'supplement', 'kitchen', 'cooking', 'recipe', 'meal'],
                'exact_matches': ['coffee beans', 'green tea', 'protein powder']
            }
        ]
        
        # First check for exact matches
        for dataset in datasets:
            for exact_match in dataset.get('exact_matches', []):
                if exact_match in product_query_lower:
                    return dataset
        
        # Then check for word-boundary keyword matches
        import re
        for dataset in datasets:
            for keyword in dataset['keywords']:
                # Use word boundaries to ensure exact word matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, product_query_lower):
                    return dataset
        
        # Default to Amazon products dataset for electronics/general products
        return datasets[0]

    def analyze_single_review(self, review_text: str) -> Tuple[str, str, Optional[gr.Plot]]:
        """Analyze a single customer review"""
        if not review_text.strip():
            error_msg = "❌ Please enter a customer review to analyze."
            return error_msg, error_msg, None
        
        try:
            print(f"🔍 Analyzing single review...")
            print(f"📝 Review length: {len(review_text)} characters")
            
            # Use the analyze_customer_feedback method
            result = self.summarizer.analyze_customer_feedback([review_text])
            
            if not result or 'error' in result:
                error_msg = "❌ Analysis failed. Please check your input and try again."
                return error_msg, error_msg, None
            
            # Extract data from result
            sentiment_data = result.get('sentiment', {})
            summary_data = result.get('summary', {})
            
            # Format outputs
            sentiment_result = self._format_sentiment_output_single(sentiment_data, review_text)
            summary_result = self._format_summary_output(summary_data)
            
            # Create sentiment chart for single review
            chart = self._create_single_review_chart(result.get('sentiment', {}))
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, None

    def analyze_multiple_reviews(self, reviews_text: str) -> Tuple[str, str, Optional[gr.Plot]]:
        """Analyze multiple customer reviews"""
        if not reviews_text.strip():
            error_msg = "❌ Please enter some customer reviews to analyze."
            return error_msg, error_msg, None
        
        try:
            print(f"📊 Analyzing multiple reviews...")
            
            # Split reviews by double newlines or single newlines
            if '\n\n' in reviews_text:
                reviews = [r.strip() for r in reviews_text.split('\n\n') if r.strip()]
            else:
                reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
            
            if not reviews:
                error_msg = "❌ No valid reviews found. Please check your input format."
                return error_msg, error_msg, None
            
            print(f"📝 Input text length: {len(reviews_text)} characters")
            
            # Use the analyze_customer_feedback method which handles multiple reviews
            result = self.summarizer.analyze_customer_feedback(reviews)
            
            if not result or 'error' in result:
                error_msg = "❌ Analysis failed. Please check your input and try again."
                return error_msg, error_msg, None
            
            # Extract data from result
            sentiment_data = result.get('sentiment', {})
            summary_data = result.get('summary', {})
            
            # Format outputs
            sentiment_result = self._format_sentiment_output_multiple(sentiment_data, len(reviews))
            summary_result = self._format_summary_output(summary_data)
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(sentiment_data)
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, None

    def _format_sentiment_output_single(self, sentiment_data: Dict, review_text: str) -> str:
        """Format sentiment analysis output for a single review"""
        if not sentiment_data:
            return "❌ No sentiment analysis results available."
        
        # Get the prediction for the single review
        overall_sentiment = sentiment_data.get('overall_sentiment', 'Unknown')
        confidence = sentiment_data.get('confidence', 0)
        
        # Get individual scores
        scores = sentiment_data.get('scores', {})
        
        output = f"🎯 **Sentiment Analysis Results**\n\n"
        output += f"**Overall Sentiment:** {overall_sentiment}\n"
        output += f"**Confidence:** {confidence:.1%}\n\n"
        
        if scores:
            output += f"**Detailed Scores:**\n"
            for sentiment, score in scores.items():
                percentage = score * 100
                bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
                output += f"• {sentiment.title()}: {percentage:.1f}% {bar}\n"
        
        # Add review preview
        preview = review_text[:200] + "..." if len(review_text) > 200 else review_text
        output += f"\n**Review Preview:**\n*{preview}*"
        
        return output

    def _format_sentiment_output_multiple(self, sentiment_data: Dict, review_count: int) -> str:
        """Format sentiment analysis output for multiple reviews"""
        if not sentiment_data:
            return "❌ No sentiment analysis results available."
        
        overall_sentiment = sentiment_data.get('overall_sentiment', 'Unknown')
        confidence = sentiment_data.get('confidence', 0)
        distribution = sentiment_data.get('distribution', {})
        
        output = f"📊 **Sentiment Analysis Results**\n\n"
        output += f"**Reviews Analyzed:** {review_count}\n"
        output += f"**Overall Sentiment:** {overall_sentiment}\n"
        output += f"**Confidence:** {confidence:.1%}\n\n"
        
        if distribution:
            output += f"**Sentiment Distribution:**\n"
            for sentiment, percentage in distribution.items():
                count = int(percentage * review_count)
                bar = "█" * int(percentage * 20) + "░" * (20 - int(percentage * 20))
                output += f"• {sentiment.title()}: {percentage:.1%} ({count} reviews) {bar}\n"
        
        return output

    def _format_summary_output(self, summary_data: Dict) -> str:
        """Format summary output"""
        if not summary_data:
            return "❌ No summary data available."
        
        # Check if there's an error
        if summary_data.get('error'):
            return f"❌ Summary generation failed: {summary_data.get('error')}"
        
        # Get summary text
        summary_text = summary_data.get('text', '')
        
        # If no text but not explicitly failed, try alternative fields
        if not summary_text:
            summary_text = summary_data.get('summary', '')
        
        if not summary_text or summary_text == 'No summary available':
            return "⚠️ Summary could not be generated for this content."
        
        output = f"📝 **AI-Generated Summary & Insights**\n\n"
        output += f"{summary_text}\n\n"
        
        # Add technical details
        if summary_data.get('compression_ratio') and summary_data['compression_ratio'] > 0:
            compression = summary_data['compression_ratio']
            output += f"**Summary Stats:**\n"
            output += f"• Compression Ratio: {compression:.1%}\n"
        
        # Add model info if available
        if summary_data.get('model_used'):
            output += f"• Model Used: {summary_data['model_used']}\n"
        
        return output

    def _create_sentiment_distribution_chart(self, sentiment_data: Dict) -> Optional[gr.Plot]:
        """Create a sentiment distribution chart"""
        try:
            distribution = sentiment_data.get('distribution', {})
            if not distribution:
                return None
            
            # Prepare data
            sentiments = list(distribution.keys())
            values = list(distribution.values())
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color mapping for sentiments
            colors = {
                'positive': '#4CAF50',
                'negative': '#F44336', 
                'neutral': '#FF9800'
            }
            
            bar_colors = [colors.get(sentiment.lower(), '#757575') for sentiment in sentiments]
            
            bars = ax.bar(sentiments, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Customize the plot
            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Sentiment', fontsize=12)
            ax.set_ylabel('Percentage', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.1%}',
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Improve appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None

    def _create_single_review_chart(self, sentiment_data: Dict) -> Optional[gr.Plot]:
        """Create a chart for single review sentiment scores"""
        try:
            scores = sentiment_data.get('scores', {})
            if not scores:
                return None
            
            # Prepare data
            sentiments = list(scores.keys())
            values = list(scores.values())
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Color mapping
            colors = {
                'positive': '#4CAF50',
                'negative': '#F44336',
                'neutral': '#FF9800'
            }
            
            bar_colors = [colors.get(sentiment.lower(), '#757575') for sentiment in sentiments]
            
            bars = ax.bar(sentiments, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Customize the plot
            ax.set_title('Sentiment Scores', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sentiment Type', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1%}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Improve appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
    
    def clear_outputs(self):
        """Clear all outputs"""
        return "", "", "", None
    
    def refresh_history(self):
        """Refresh analysis history (placeholder)"""
        return "🔄 History refreshed! (Feature coming soon)"
    
    def create_interface(self):
        """Create the Gradio interface with bright, modern styling"""
        
        # Enhanced CSS for Ocean Blue and Coral theme
        css = """
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2196f3 0%, #21cbf3 100%);
            min-height: 100vh;
        }
        
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #ff7043 0%, #ffab40 50%, #ffcc02 100%);
            color: #1a1a1a;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.8;
            margin: 0;
        }
        
        /* Tab styling */
        .tab-nav {
            background: linear-gradient(135deg, #81c784 0%, #4fc3f7 100%);
            border-radius: 15px;
            padding: 5px;
            margin-bottom: 20px;
        }
        
        /* Card-like containers */
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        /* Button styling */
        .btn-primary {
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            color: white;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.6);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #ff7043 0%, #ff5722 100%);
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            color: white;
            box-shadow: 0 4px 15px rgba(255, 112, 67, 0.3);
        }
        
        /* Input styling */
        .input-field {
            border-radius: 12px;
            border: 2px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.9);
            padding: 12px;
            font-size: 16px;
        }
        
        .input-field:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Results styling */
        .results-container {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            color: #1565c0;
            border-left: 5px solid #2196f3;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
        }
        
        .sentiment-positive {
            background: linear-gradient(135deg, #c8e6c9 0%, #81c784 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: #2e7d32;
        }
        
        .sentiment-negative {
            background: linear-gradient(135deg, #ffcdd2 0%, #ef5350 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        
        /* Chart containers */
        .chart-container {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #ff9800;
            box-shadow: 0 4px 15px rgba(255, 152, 0, 0.1);
        }
        
        /* Guide styling */
        .guide-section {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            color: #2d3748;
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            .main-header p {
                font-size: 1rem;
            }
        }
        """
        
        with gr.Blocks(css=css, title="Customer Sentiment Summarizer", theme=gr.themes.Soft()) as interface:
            
            # Enhanced Header
            gr.Markdown("""
            <div class="main-header">
                <h1>🌟 Customer Sentiment Summarizer 🌟</h1>
                <p>✨ Powered by Advanced AI Models: BERT & BART ✨</p>
                <p>🚀 Smart Product-Based Review Analysis Platform 🚀</p>
            </div>
            """)
            
            # Welcome message with bright styling
            gr.Markdown("""
            <div class="card">
                <h2 style="color: #667eea; text-align: center; margin-bottom: 15px;">
                    🎯 Welcome to the Future of Review Analysis! 🎯
                </h2>
                <p style="text-align: center; font-size: 1.1rem; color: #4a5568;">
                    Discover what customers really think about any product with our AI-powered sentiment analysis.
                    Just enter a product name and watch the magic happen! ✨
                </p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Product Review Analysis
                with gr.TabItem("🔍 Product Review Analysis"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">
                            🔍 Smart Product-Based Review Analysis
                        </h3>
                        <p style="color: #4a5568; font-size: 1.1rem;">
                            Enter any product name to instantly analyze thousands of real customer reviews! 
                            Our AI will find the most relevant datasets and provide comprehensive insights. 🎯
                        </p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            product_search_input = gr.Textbox(
                                label="🔍 Product Search",
                                placeholder="✨ Try: iPhone 15, MacBook Pro, Samsung Galaxy, AirPods, or any product!",
                                value="iPhone",
                                info="🎯 Enter any product name and we'll find the perfect dataset for analysis!",
                                elem_classes=["input-field"]
                            )
                        with gr.Column(scale=1):
                            sample_size_input = gr.Slider(
                                label="📊 Sample Size",
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                                info="🔢 Number of reviews to analyze (more = better insights!)"
                            )
                    
                    with gr.Row():
                        product_analyze_btn = gr.Button(
                            "� Find & Analyze Reviews", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        product_clear_btn = gr.Button(
                            "🗑️ Clear Results", 
                            variant="secondary",
                            elem_classes=["btn-secondary"]
                        )
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
                        <h4 style="color: #1a202c; margin-bottom: 15px;">💡 How Our Magic Works:</h4>
                        <ul style="color: #2d3748; font-size: 1rem;">
                            <li>🔍 <strong>Smart Search:</strong> Enter any product name or category</li>
                            <li>🤖 <strong>AI Dataset Selection:</strong> We automatically find the best review dataset</li>
                            <li>📊 <strong>Real Reviews:</strong> Analyze genuine customer feedback from Amazon & other sources</li>
                            <li>🎯 <strong>Instant Insights:</strong> Get comprehensive sentiment analysis in seconds</li>
                        </ul>
                        
                         d
                       
                    </div>
                    """)
                    
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            product_sentiment_output = gr.Markdown(
                                label="Sentiment Analysis",
                                elem_classes=["results-container"]
                            )
                        with gr.Column():
                            product_summary_output = gr.Markdown(
                                label="AI Summary & Insights",
                                elem_classes=["results-container"]
                            )
                    
                    with gr.Row():
                        product_chart_output = gr.Plot(
                            label="Sentiment Distribution",
                            elem_classes=["chart-container"]
                        )
                
                # Tab 2: Single Review Analysis
                with gr.TabItem("📝 Single Review Analysis"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">
                            📝 Deep Dive into Individual Reviews
                        </h3>
                        <p style="color: #4a5568; font-size: 1.1rem;">
                            Paste any customer review to get detailed sentiment analysis with confidence scores! 
                            Perfect for understanding specific feedback in detail. 🔍
                        </p>
                    </div>
                    """)
                    
                    with gr.Row():
                        single_input = gr.Textbox(
                            label="📝 Customer Review",
                            placeholder="✨ Paste any customer review here! Try: 'This iPhone is amazing! Great camera and battery life.'",
                            lines=5,
                            elem_classes=["input-field"]
                        )
                    
                    with gr.Row():
                        single_analyze_btn = gr.Button(
                            "🔍 Analyze Review", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        single_clear_btn = gr.Button(
                            "🗑️ Clear", 
                            variant="secondary",
                            elem_classes=["btn-secondary"]
                        )
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);">
                        <h4 style="color: #1b5e20; margin-bottom: 15px;">✨ What You'll Get:</h4>
                        <ul style="color: #2e7d32; font-size: 1rem;">
                            <li>🎯 <strong>Sentiment Classification:</strong> Positive, Negative, or Neutral</li>
                            <li>📊 <strong>Confidence Scores:</strong> How certain the AI is about its prediction</li>
                            <li>📈 <strong>Detailed Breakdown:</strong> Positive vs negative percentages</li>
                            <li>🎨 <strong>Visual Chart:</strong> Beautiful sentiment visualization</li>
                        </ul>
                    </div>
                    """)
                    
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            single_sentiment_output = gr.Markdown(
                                label="Sentiment Analysis",
                                elem_classes=["results-container"]
                            )
                        with gr.Column():
                            single_summary_output = gr.Markdown(
                                label="Key Insights",
                                elem_classes=["results-container"]
                            )
                    
                    with gr.Row():
                        single_chart_output = gr.Plot(
                            label="Sentiment Scores",
                            elem_classes=["chart-container"]
                        )
                
                # Tab 3: Multiple Reviews Analysis
                with gr.TabItem("📋 Multiple Reviews Analysis"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">
                            📋 Bulk Review Analysis Powerhouse
                        </h3>
                        <p style="color: #4a5568; font-size: 1.1rem;">
                            Paste multiple customer reviews to get comprehensive insights! Perfect for 
                            analyzing batches of feedback, survey responses, or multiple product reviews. 🚀
                        </p>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                        <h4 style="color: #1a202c; margin-bottom: 15px;">✨ Pro Tips for Best Results:</h4>
                        <ul style="color: #2d3748; font-size: 1rem;">
                            <li>📝 <strong>One review per line:</strong> Separate each review with a new line</li>
                            <li>📋 <strong>Mixed content OK:</strong> Positive and negative reviews together</li>
                            <li>🎯 <strong>Any length:</strong> Short tweets to long detailed reviews</li>
                            <li>⚡ <strong>Fast processing:</strong> Handles hundreds of reviews quickly</li>
                        </ul>
                    </div>
                    """)
                    
                    multiple_input = gr.Textbox(
                        label="📝 Multiple Customer Reviews",
                        placeholder="✨ Paste multiple reviews here, one per line:\n\nThis iPhone is absolutely fantastic! Love the camera quality.\nNot impressed with the battery life, dies too quickly.\nGreat phone overall, very happy with my purchase!\nThe screen is beautiful but it's quite expensive.\nBest phone I've ever owned, highly recommend!",
                        lines=12,
                        elem_classes=["input-field"]
                    )
                    
                    with gr.Row():
                        multiple_analyze_btn = gr.Button(
                            "� Analyze All Reviews", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        multiple_clear_btn = gr.Button(
                            "🗑️ Clear All", 
                            variant="secondary",
                            elem_classes=["btn-secondary"]
                        )
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
                        <h4 style="color: #1a202c; margin-bottom: 15px;">📊 Comprehensive Analytics You'll Get:</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; color: #2d3748;">
                            <div>
                                <strong>📈 Overall Sentiment:</strong> Positive/negative/neutral breakdown
                            </div>
                            <div>
                                <strong>🎯 Individual Scores:</strong> Each review analyzed separately
                            </div>
                            <div>
                                <strong>📊 Visual Charts:</strong> Beautiful sentiment distribution graphs
                            </div>
                            <div>
                                <strong>🤖 Smart Summary:</strong> AI-generated key insights and trends
                            </div>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("### 📊 Comprehensive Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            multiple_sentiment_output = gr.Markdown(
                                label="📈 Sentiment Analysis",
                                elem_classes=["results-container"]
                            )
                        with gr.Column():
                            multiple_summary_output = gr.Markdown(
                                label="🤖 AI Summary",
                                elem_classes=["results-container"]
                            )
                    
                    with gr.Row():
                        multiple_chart_output = gr.Plot(
                            label="📊 Sentiment Distribution",
                            elem_classes=["chart-container"]
                        )
                
                # Tab 4: Quick Guide
                with gr.TabItem("📖 Quick Guide"):
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); color: white;">
                        <h2 style="color: white; margin-bottom: 20px;">
                            🚀 Welcome to Your AI-Powered Sentiment Analysis Platform!
                        </h2>
                        <p style="font-size: 1.2rem; color: #1a202c;">
                            Transform customer feedback into actionable insights with cutting-edge AI technology. 
                            Get started in seconds! ✨
                        </p>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #ff7043 0%, #ff5722 100%);">
                        <h3 style="color: white; margin-bottom: 15px;">🔍 Product Review Analysis (🌟 Primary Feature)</h3>
                        <div style="color: #1a202c;">
                            <p><strong>🎯 Perfect for:</strong> Large-scale analysis of thousands of customer reviews</p>
                            <p><strong>⚡ How to use:</strong> Simply enter any product name like "iPhone", "MacBook", "Tesla"</p>
                            <p><strong>🚀 Features:</strong> Automated dataset selection, smart sampling, comprehensive AI analysis</p>
                            <p><strong>📊 Sample sizes:</strong> Start with 50-100 for quick tests, use 200+ for detailed insights</p>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);">
                        <h4 style="color: white; margin-bottom: 15px;">🎯 Smart Product Search Engine:</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; color: #1a202c;">
                            <div>
                                <strong>📱 Electronics:</strong><br>
                                iPhone, Samsung, MacBook, iPad → Amazon Product Reviews (28K+ reviews)
                            </div>
                            <div>
                                <strong>🍕 Food & Beverages:</strong><br>
                                Coffee, snacks, organic foods → Amazon Fine Food Reviews
                            </div>
                            <div>
                                <strong>🏠 Home & Garden:</strong><br>
                                Furniture, appliances → Amazon Product Reviews
                            </div>
                            <div>
                                <strong>🎮 Everything Else:</strong><br>
                                Games, books, toys → Amazon Product Reviews (default)
                            </div>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
                        <h3 style="color: #2d3748; margin-bottom: 15px;">📝 Single Review Deep Dive</h3>
                        <div style="color: #4a5568;">
                            <p><strong>🎯 Perfect for:</strong> Detailed analysis of individual customer reviews</p>
                            <p><strong>⚡ How to use:</strong> Copy any review from Amazon, Google, social media, or surveys</p>
                            <p><strong>🚀 Features:</strong> Detailed sentiment scores, confidence levels, and insights</p>
                            <p><strong>💡 Pro tip:</strong> Great for understanding specific customer pain points!</p>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #a6c1ee 0%, #fbc2eb 100%);">
                        <h3 style="color: #2d3748; margin-bottom: 15px;">📋 Bulk Review Analytics</h3>
                        <div style="color: #4a5568;">
                            <p><strong>🎯 Perfect for:</strong> Analyzing your own review collections and survey responses</p>
                            <p><strong>⚡ How to use:</strong> Paste multiple reviews, one per line or separated by blank lines</p>
                            <p><strong>🚀 Features:</strong> Overall sentiment distribution, statistics, and AI-powered summary</p>
                            <p><strong>📊 Great for:</strong> Survey analysis, feedback batches, competitive research</p>
                        </div>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                        <h3 style="color: white; margin-bottom: 15px;">💡 Pro Tips for Amazing Results</h3>
                        <ul style="color: #1a202c; font-size: 1.1rem;">
                            <li><strong>🔍 Product Search:</strong> Use specific names ("iPhone 15 Pro") or categories ("gaming laptop")</li>
                            <li><strong>📝 Manual Analysis:</strong> Include complete sentences for the most accurate sentiment detection</li>
                            <li><strong>📊 Sample Sizes:</strong> Start with 50 reviews for quick insights, scale to 200+ for comprehensive analysis</li>
                            <li><strong>🎯 Confidence Scores:</strong> Higher scores (>80%) indicate more reliable predictions</li>
                            <li><strong>⚡ Speed:</strong> Processing is lightning-fast - most analyses complete in under 30 seconds!</li>
                        </ul>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    <div style="text-align: center; margin-top: 30px;">
                        <h2 style="color: #667eea;">🎉 Ready to Transform Your Customer Insights?</h2>
                        <p style="color: #4a5568; font-size: 1.2rem;">
                            Start with the <strong>Product Review Analysis</strong> tab above! 
                            Just enter any product name and watch the magic happen ✨
                        </p>
                    </div>
                    """)
            
            # Event handlers
            
            # Product analysis tab
            product_analyze_btn.click(
                fn=self.analyze_product_reviews,
                inputs=[product_search_input, sample_size_input],
                outputs=[product_sentiment_output, product_summary_output, product_chart_output]
            )
            
            product_clear_btn.click(
                fn=self.clear_outputs,
                outputs=[product_search_input, product_sentiment_output, product_summary_output, product_chart_output]
            )
            
            # Single review tab
            single_analyze_btn.click(
                fn=self.analyze_single_review,
                inputs=[single_input],
                outputs=[single_sentiment_output, single_summary_output, single_chart_output]
            )
            
            single_clear_btn.click(
                fn=self.clear_outputs,
                outputs=[single_input, single_sentiment_output, single_summary_output, single_chart_output]
            )
            
            # Multiple reviews tab
            multiple_analyze_btn.click(
                fn=self.analyze_multiple_reviews,
                inputs=[multiple_input],
                outputs=[multiple_sentiment_output, multiple_summary_output, multiple_chart_output]
            )
            
            multiple_clear_btn.click(
                fn=self.clear_outputs,
                outputs=[multiple_input, multiple_sentiment_output, multiple_summary_output, multiple_chart_output]
            )
        
        return interface

def main():
    """Main function to run the Gradio app"""
    try:
        # Initialize the app
        app = SentimentApp()
        
        # Create and launch the interface
        interface = app.create_interface()
        
        # Launch the app
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except Exception as e:
        print(f"❌ Error starting the application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

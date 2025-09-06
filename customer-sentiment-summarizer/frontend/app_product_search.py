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
        
        # 🌟 BRIGHT & CLEAN MODERN THEME 🌟
        css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {
            box-sizing: border-box;
        }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
            min-height: 100vh;
            color: #1e293b;
        }
        
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            color: #1e293b;
            padding: 40px;
            border-radius: 20px;
            margin: 20px;
            box-shadow: 
                0 10px 25px rgba(0, 0, 0, 0.08),
                0 4px 10px rgba(0, 0, 0, 0.04);
            border: 1px solid rgba(226, 232, 240, 0.8);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
            border-radius: 20px 20px 0 0;
        }
        
        .main-header h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 15px;
            color: #1e293b;
            letter-spacing: -1px;
        }
        
        .main-header p {
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            color: #475569;
            margin: 10px 0;
            font-weight: 400;
        }
        
        /* Clean card styling */
        .card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            padding: 32px;
            margin: 24px 0;
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 
                0 4px 6px rgba(0, 0, 0, 0.05),
                0 1px 3px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            color: #1e293b;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 8px 15px rgba(0, 0, 0, 0.1),
                0 3px 6px rgba(0, 0, 0, 0.08);
        }
        
        /* Modern button styling */
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            border-radius: 12px;
            padding: 16px 32px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #ffffff;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 14px rgba(59, 130, 246, 0.3),
                0 2px 4px rgba(0, 0, 0, 0.1);
            cursor: pointer;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 6px 20px rgba(59, 130, 246, 0.4),
                0 4px 8px rgba(0, 0, 0, 0.15);
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
            border: none;
            border-radius: 12px;
            padding: 16px 32px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #ffffff;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 14px rgba(107, 114, 128, 0.3),
                0 2px 4px rgba(0, 0, 0, 0.1);
            cursor: pointer;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 6px 20px rgba(107, 114, 128, 0.4),
                0 4px 8px rgba(0, 0, 0, 0.15);
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        }
        
        /* Clean input styling */
        .input-field {
            background: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 16px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 16px !important;
            color: #1e293b !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        .input-field:focus {
            border-color: #3b82f6 !important;
            box-shadow: 
                0 0 0 3px rgba(59, 130, 246, 0.1) !important,
                0 2px 8px rgba(0, 0, 0, 0.1) !important;
            outline: none !important;
        }
        
        .input-field::placeholder {
            color: #94a3b8 !important;
            font-style: normal !important;
        }
        
        /* Clean results containers */
        .results-container {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 16px;
            padding: 28px;
            margin: 20px 0;
            color: #1e293b;
            border: 1px solid rgba(14, 165, 233, 0.2);
            box-shadow: 
                0 4px 6px rgba(14, 165, 233, 0.05),
                0 1px 3px rgba(0, 0, 0, 0.1);
            font-family: 'Inter', sans-serif;
            font-size: 16px;
            line-height: 1.6;
        }
        
        /* Chart containers */
        .chart-container {
            background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
            border-radius: 16px;
            padding: 28px;
            margin: 20px 0;
            border: 1px solid rgba(245, 158, 11, 0.2);
            box-shadow: 
                0 4px 6px rgba(245, 158, 11, 0.05),
                0 1px 3px rgba(0, 0, 0, 0.1);
            color: #1e293b;
        }
        
        /* Tab styling */
        .tab-nav {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            padding: 8px;
            margin-bottom: 24px;
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Typography improvements */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
        }
        
        h2 {
            color: #334155 !important;
        }
        
        h3 {
            color: #475569 !important;
        }
        
        p, span, div, li {
            font-family: 'Inter', sans-serif !important;
            color: #334155 !important;
            font-weight: 400 !important;
            line-height: 1.6 !important;
        }
        
        strong {
            color: #1e293b !important;
            font-weight: 600 !important;
        }
        
        /* Accent colors for highlights */
        .accent-blue {
            color: #3b82f6 !important;
            font-weight: 600 !important;
        }
        
        .accent-purple {
            color: #8b5cf6 !important;
            font-weight: 600 !important;
        }
        
        .accent-pink {
            color: #ec4899 !important;
            font-weight: 600 !important;
        }
        
        .accent-amber {
            color: #f59e0b !important;
            font-weight: 600 !important;
        }
        
        /* Special info boxes */
        .info-box {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border: 1px solid rgba(14, 165, 233, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            color: #1e293b;
        }
        
        .success-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            color: #1e293b;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            color: #1e293b;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2.2rem;
            }
            .main-header p {
                font-size: 1.1rem;
            }
            .card {
                padding: 24px;
                margin: 16px 0;
            }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 6px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #cbd5e1, #94a3b8);
            border-radius: 6px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #94a3b8, #64748b);
        }
        """
        
        with gr.Blocks(css=css, title="Customer Sentiment Summarizer", theme=gr.themes.Soft()) as interface:
            
            # ✨ CLEAN & BRIGHT HEADER ✨
            gr.Markdown("""
            <div class="main-header">
                <h1>🎯 Customer Sentiment Analyzer</h1>
                <p><span class="accent-blue">Powered by Advanced AI:</span> <span class="accent-purple">BERT</span> & <span class="accent-pink">BART</span></p>
                <p><span class="accent-amber">Smart Product Review Analysis Platform</span></p>
            </div>
            """)
            
            # CLEAN WELCOME INTERFACE
            gr.Markdown("""
            <div class="card">
                <h2 style="text-align: center; margin-bottom: 20px;">
                    🌟 Welcome to Professional Review Analysis
                </h2>
                <div style="text-align: center; font-size: 1.1rem; line-height: 1.8;">
                    <p><strong class="accent-blue">Discover customer insights</strong> with our AI-powered sentiment analysis</p>
                    <p><strong class="accent-purple">Advanced machine learning</strong> models process thousands of reviews instantly</p>
                    <p><strong class="accent-pink">Beautiful visualizations</strong> and comprehensive summaries at your fingertips</p>
                    <div class="info-box" style="margin-top: 20px;">
                        <strong>🚀 Ready to start?</strong> Enter any product name above to analyze real customer reviews!
                    </div>
                </div>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # MATRIX SCANNER TAB
                with gr.TabItem("� QUANTUM MATRIX SCANNER"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="margin-bottom: 20px; font-size: 1.8rem;">
                            � NEURAL QUANTUM MATRIX EMOTION SCANNER
                        </h3>
                        <p style="font-size: 1.2rem; line-height: 1.6;">
                            🌌 <strong>INITIATE MATRIX SCAN:</strong> Upload any product designation into our quantum neural grid! 
                            Our cybernetic emotion-analysis cores will scan through thousands of digital consciousness fragments 
                            to decode the collective emotional matrix! 🤖✨
                        </p>
                        <div style="margin-top: 15px; padding: 15px; background: rgba(255, 0, 255, 0.1); border-radius: 10px; border-left: 4px solid #ff00ff;">
                            <strong style="color: #ff00ff;">⚡ QUANTUM PROCESS:</strong>
                            <span style="color: #00ffff;">Neural networks → Emotion extraction → Matrix compilation → Holographic visualization</span>
                        </div>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            product_search_input = gr.Textbox(
                                label="� QUANTUM PRODUCT MATRIX",
                                placeholder="⚡ ENTER MATRIX CODE: iPhone-15 | MacBook-Pro | Galaxy-Ultra | AirPods-Max | Tesla-Cybertruck ⚡",
                                value="iPhone",
                                info="� INPUT ANY PRODUCT DESIGNATION FOR NEURAL MATRIX SCAN 🌐",
                                elem_classes=["input-field"]
                            )
                        with gr.Column(scale=1):
                            sample_size_input = gr.Slider(
                                label="🧠 NEURAL SAMPLE MATRIX",
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                                info="⚡ QUANTUM PROCESSING POWER (More = Deeper Matrix Scan)"
                            )
                    
                    with gr.Row():
                        product_analyze_btn = gr.Button(
                            "🚀 INITIATE MATRIX SCAN", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        product_clear_btn = gr.Button(
                            "� NEURAL RESET", 
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
                
                # NEURAL SINGLE-UNIT ANALYSIS
                with gr.TabItem("🧠 NEURAL UNIT SCANNER"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="margin-bottom: 20px; font-size: 1.8rem;">
                            🧠 INDIVIDUAL CONSCIOUSNESS PROBE
                        </h3>
                        <p style="font-size: 1.2rem; line-height: 1.6;">
                            🔬 <strong>SINGLE NEURAL MATRIX ANALYSIS:</strong> Upload any singular digital consciousness fragment! 
                            Our advanced emotion-detection algorithms will perform a deep neural scan to decode 
                            the complete emotional spectrum with quantum-level precision! ⚡🎯
                        </p>
                        <div style="margin-top: 15px; padding: 15px; background: rgba(0, 255, 255, 0.1); border-radius: 10px; border-left: 4px solid #00ffff;">
                            <strong style="color: #00ffff;">🔬 PRECISION MODE:</strong>
                            <span style="color: #ff00ff;">Deep neural scanning → Emotion mapping → Confidence calculation → Holographic display</span>
                        </div>
                    </div>
                    """)
                    
                    with gr.Row():
                        single_input = gr.Textbox(
                            label="🧠 CONSCIOUSNESS FRAGMENT INPUT",
                            placeholder="⚡ INSERT NEURAL DATA: 'This iPhone is a technological masterpiece! Neural pathways activated by superior camera matrix and quantum battery core.' ⚡",
                            lines=5,
                            elem_classes=["input-field"]
                        )
                    
                    with gr.Row():
                        single_analyze_btn = gr.Button(
                            "� NEURAL PROBE ACTIVATED", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        single_clear_btn = gr.Button(
                            "🧪 CLEAR PROBE", 
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
                
                # MULTI-CONSCIOUSNESS COLLECTIVE ANALYSIS
                with gr.TabItem("🌌 HIVE MIND ANALYZER"):
                    gr.Markdown("""
                    <div class="card">
                        <h3 style="margin-bottom: 20px; font-size: 1.8rem;">
                            🌌 COLLECTIVE CONSCIOUSNESS MATRIX
                        </h3>
                        <p style="font-size: 1.2rem; line-height: 1.6;">
                            💫 <strong>HIVE MIND NEURAL NETWORK:</strong> Upload multiple digital consciousness fragments 
                            for collective emotion matrix analysis! Our quantum processors will synchronize with 
                            the hive mind to extract patterns from the collective digital soul! 🔮⚡
                        </p>
                        <div style="margin-top: 15px; padding: 15px; background: rgba(255, 102, 0, 0.1); border-radius: 10px; border-left: 4px solid #ff6600;">
                            <strong style="color: #ff6600;">🌟 COLLECTIVE MODE:</strong>
                            <span style="color: #00ffff;">Mass consciousness upload → Hive mind sync → Pattern recognition → Collective sentiment hologram</span>
                        </div>
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

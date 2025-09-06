#!/usr/bin/env python3
"""
Enhanced Customer Sentiment Summarizer with Multiple Review Collection Methods
This is a simplified version that bypasses Gradio file upload issues
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
        pri                # Tab 1: Product Review Analysis
                with gr.TabItem("🔍 Product Review Analysis"):("🚀 Starting Customer Sentiment Summarizer Chatbot...")
        print("🚀 Initializing AI models...")
        
        # Initialize the main summarizer
        self.summarizer = CustomerSentimentSummarizer()
        
        print("✅ AI models loaded successfully!")
        
        # Set up matplotlib for better chart appearance
        plt.style.use('default')
        sns.set_palette("husl")
    
    def scrape_and_analyze(self, product_name: str) -> Tuple[str, str, str, Optional[gr.Plot]]:
        """Scrape reviews and perform sentiment analysis"""
        if not product_name.strip():
            error_msg = "❌ Please enter a product name to search for."
            return error_msg, error_msg, error_msg, None
        
        try:
            print(f"🕷️ Scraping reviews for: {product_name}")
            
            # Start timing
            start_time = time.time()
            
            # Use the summarizer to scrape and analyze
            result = self.summarizer.scrape_and_analyze_reviews(product_name, max_reviews=20)
            
            # Add timing information
            result.update({
                'total_time': time.time() - start_time
            })
            
            # Format results
            scraping_result = self._format_scraping_output(result)
            
            # Get analysis results from the nested structure
            analysis = result.get('analysis', {})
            sentiment_result = self._format_sentiment_output_multiple(
                analysis.get('sentiment', {}), 
                result.get('reviews_found', 0)
            )
            summary_result = self._format_summary_output(analysis.get('summary', {}))
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(analysis.get('sentiment', {}))
            
            return scraping_result, sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during scraping and analysis: {str(e)}"
            return error_msg, error_msg, error_msg, None
    
    def _format_scraping_output(self, result: Dict) -> str:
        """Format scraping results output"""
        if not result.get('success'):
            output = f"❌ **Scraping Failed**\n\n"
            
            error = result.get('error', 'Unknown error')
            output += f"**Error:** {error}\n\n"
            
            output += "**💡 Try This Instead:**\n"
            output += "1. **Manual Analysis**: Use the 'Single Review Analysis' or 'Multiple Reviews Analysis' tabs\n"
            output += "2. **Copy & Paste**: Find reviews on Amazon/other sites and paste them directly\n"
            output += "3. **Different Product**: Try searching for a different product name\n\n"
            output += "The manual analysis features provide the same powerful insights!"
            
            return output
        
        output = f"🕷️ **Web Scraping Results**\n\n"
        output += f"**Product:** {result.get('product_name', 'Unknown')}\n"
        output += f"**Reviews Found:** {result.get('reviews_found', 0)}\n"
        output += f"**Scraping Time:** {result.get('scraping_time', 0):.1f} seconds\n"
        output += f"**Analysis Time:** {result.get('analysis_time', 0):.1f} seconds\n"
        output += f"**Total Time:** {result.get('total_time', 0):.1f} seconds\n\n"
        
        # Show sample reviews
        reviews = result.get('sample_reviews', [])
        if reviews:
            output += "**📝 Sample Reviews Found:**\n"
            for i, review in enumerate(reviews[:3], 1):
                # Handle different review formats
                if isinstance(review, dict):
                    review_text = review.get('text', review.get('review', str(review)))
                else:
                    review_text = str(review)
                # Truncate long reviews
                if len(review_text) > 100:
                    review_text = review_text[:100] + "..."
                output += f"{i}. {review_text}\n"
            
            if len(reviews) > 3:
                output += f"... and {len(reviews) - 3} more reviews\n"
        
        return output
    
    def analyze_single_review(self, review_text: str) -> Tuple[str, str, Optional[gr.Plot]]:
        """Analyze a single review"""
        if not review_text.strip():
            error_msg = "❌ Please enter a review to analyze."
            return error_msg, error_msg, None
        
        try:
            print(f"🔍 Analyzing single review...")
            
            # Analyze the review
            result = self.summarizer.analyze_customer_feedback([review_text])
            
            # Format results
            sentiment_result = self._format_sentiment_output_single(
                result.get('sentiment', {}), 
                review_text
            )
            summary_result = self._format_summary_output(result.get('summary', {}))
            
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
            print(f"📝 Input text length: {len(reviews_text)} characters")
            
            # Use the analyze_customer_feedback method which handles multiple reviews
            result = self.summarizer.analyze_customer_feedback(reviews_text)
            
            if not result or 'error' in result:
                error_msg = "❌ Analysis failed. Please check your input and try again."
                return error_msg, error_msg, None
            
            # Extract data from result
            sentiment_data = result.get('sentiment', {})
            summary_data = result.get('summary', {})
            
            # Format outputs
            sentiment_result = self._format_sentiment_output_multiple(sentiment_data, result.get('total_reviews', 1))
            summary_result = self._format_summary_output(summary_data)
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(sentiment_data)
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, None

    def analyze_kaggle_dataset(self, dataset_id: str, sample_size: int) -> Tuple[str, str, Optional[gr.Plot]]:
        """Analyze reviews from a Kaggle dataset"""
        if not dataset_id.strip():
            error_msg = "❌ Please enter a Kaggle dataset ID (e.g., arhamrumi/amazon-product-reviews)."
            return error_msg, error_msg, None
        
        try:
            print(f"📦 Loading Kaggle dataset: {dataset_id}")
            
            # Import the Kaggle importer
            sys.path.append(str(Path(__file__).parent.parent / 'backend'))
            from kaggle_importer import KaggleDatasetImporter
            
            # Initialize importer
            importer = KaggleDatasetImporter()
            
            # Try to load the dataset
            if 'amazon-product-reviews' in dataset_id.lower():
                reviews, ratings = importer.process_amazon_product_reviews(dataset_id)
            elif 'turkish-product-reviews' in dataset_id.lower():
                # Download and process Turkish dataset
                dataset_path = importer.download_dataset(dataset_id)
                reviews, ratings = importer.process_turkish_product_reviews(dataset_path)
            else:
                # Try generic processing
                dataset_path = importer.download_dataset(dataset_id)
                reviews, ratings = importer.load_dataset_for_analysis(dataset_path)
            
            if not reviews:
                error_msg = f"❌ No reviews found in dataset {dataset_id}. Please check the dataset ID."
                return error_msg, error_msg, None
            
            # Sample the data if too large
            if len(reviews) > sample_size:
                import random
                indices = random.sample(range(len(reviews)), sample_size)
                reviews = [reviews[i] for i in indices]
                if ratings:
                    ratings = [ratings[i] for i in indices if i < len(ratings)]
                print(f"📊 Sampled {sample_size} reviews from {len(reviews)} total")
            
            print(f"🔍 Analyzing {len(reviews)} reviews from Kaggle dataset...")
            
            # Analyze the reviews
            result = self.summarizer.analyze_customer_feedback(reviews)
            
            # Format results with dataset info
            dataset_info = f"📦 **Kaggle Dataset Analysis**\n\n"
            dataset_info += f"**Dataset:** {dataset_id}\n"
            dataset_info += f"**Reviews Analyzed:** {len(reviews)}\n"
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                dataset_info += f"**Average Rating:** {avg_rating:.1f}/5.0\n"
            dataset_info += f"**Processing Time:** {result.get('processing_time', 0):.1f} seconds\n\n"
            
            sentiment_result = dataset_info + self._format_sentiment_output_multiple(
                result.get('sentiment', {}), 
                len(reviews)
            )
            
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Add rating analysis if available
            if ratings:
                summary_result += f"\n\n⭐ **Rating Analysis:**\n"
                rating_dist = {}
                for rating in ratings:
                    rating_dist[rating] = rating_dist.get(rating, 0) + 1
                
                for rating in sorted(rating_dist.keys(), reverse=True):
                    count = rating_dist[rating]
                    percentage = (count / len(ratings)) * 100
                    stars = "★" * int(rating) + "☆" * (5 - int(rating))
                    summary_result += f"• {stars} ({rating}): {count} reviews ({percentage:.1f}%)\n"
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(result.get('sentiment', {}))
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error loading Kaggle dataset: {str(e)}\n\n"
            error_msg += "**💡 Troubleshooting:**\n"
            error_msg += "1. Check the dataset ID format (owner/dataset-name)\n"
            error_msg += "2. Ensure the dataset contains review text\n"
            error_msg += "3. Try a smaller sample size\n"
            error_msg += "4. Use manual input as alternative\n"
            return error_msg, error_msg, None
        """Analyze multiple reviews from text input"""
        if not reviews_text.strip():
            error_msg = "❌ Please enter reviews to analyze (one per line or separated by blank lines)."
            return error_msg, error_msg, None
        
        try:
            # Split reviews by double newlines or single newlines
            if '\n\n' in reviews_text:
                reviews = [r.strip() for r in reviews_text.split('\n\n') if r.strip()]
            else:
                reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
            
            if not reviews:
                error_msg = "❌ No valid reviews found. Please check your input format."
                return error_msg, error_msg, None
            
            print(f"🔍 Analyzing {len(reviews)} reviews...")
            
            # Analyze the reviews
            result = self.summarizer.analyze_customer_feedback(reviews)
            
            # Format results
            sentiment_result = self._format_sentiment_output_multiple(
                result.get('sentiment', {}), 
                len(reviews)
            )
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(result.get('sentiment', {}))
            
            return sentiment_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, None
    
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
            search_filter = dataset_mapping.get('filter', None)
            
            print(f"📦 Using dataset: {dataset_id}")
            if search_filter:
                print(f"🔍 Filtering for: {search_filter}")
            
            # Import the Kaggle importer
            sys.path.append(str(Path(__file__).parent.parent / 'backend'))
            from kaggle_importer import KaggleDatasetImporter
            
            # Initialize importer
            importer = KaggleDatasetImporter()
            
            # Load the dataset
            dataset_path = importer.download_dataset(dataset_id)
            reviews, ratings = importer.load_dataset_for_analysis(
                dataset_path,
                text_column='reviews.text',
                rating_column='reviews.rating'
            )
            
            if not reviews:
                error_msg = f"❌ No reviews found for '{product_query}'. Please try a different search term."
                return error_msg, error_msg, None
            
            # Filter reviews based on product query if needed
            if search_filter:
                filtered_reviews = []
                filtered_ratings = []
                for i, review in enumerate(reviews):
                    if review and any(keyword in str(review).lower() for keyword in search_filter):
                        filtered_reviews.append(review)
                        if ratings and i < len(ratings):
                            filtered_ratings.append(ratings[i])
                
                if filtered_reviews:
                    reviews = filtered_reviews
                    ratings = filtered_ratings
                    print(f"🎯 Found {len(reviews)} reviews matching '{product_query}'")
            
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
        """Map product search queries to appropriate datasets"""
        
        # Define dataset mappings
        datasets = [
            {
                'dataset_id': 'datafiniti/consumer-reviews-of-amazon-products',
                'description': 'Amazon Product Reviews (28K+ reviews)',
                'keywords': ['amazon', 'product', 'general', 'electronics', 'phone', 'iphone', 'samsung', 'laptop', 'computer', 'tablet', 'ipad', 'headphones', 'speaker', 'camera', 'tv', 'appliance'],
                'filter': None  # No filtering needed for general Amazon products
            },
            {
                'dataset_id': 'snap/amazon-fine-food-reviews',
                'description': 'Amazon Fine Food Reviews',
                'keywords': ['food', 'snack', 'coffee', 'tea', 'drink', 'beverage', 'nutrition', 'supplement', 'kitchen', 'cooking'],
                'filter': None
            }
        ]
        
        # Find the best matching dataset
        for dataset in datasets:
            if any(keyword in product_query for keyword in dataset['keywords']):
                return dataset
        
        # Default to Amazon products dataset
        return datasets[0]

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
        output += f"**Review:** {review_text[:200]}{'...' if len(review_text) > 200 else ''}\n\n"
        output += f"**Overall Sentiment:** {overall_sentiment.title()} ({confidence:.1%} confidence)\n\n"
        
        if scores:
            output += f"**Detailed Scores:**\n"
            for sentiment, score in scores.items():
                output += f"• {sentiment.title()}: {score:.1%}\n"
            output += "\n"
        
        # Add interpretation
        if confidence > 0.8:
            output += f"🔍 **Interpretation:** This review shows a clearly {overall_sentiment.lower()} sentiment with high confidence.\n"
        elif confidence > 0.6:
            output += f"🔍 **Interpretation:** This review leans {overall_sentiment.lower()} with moderate confidence.\n"
        else:
            output += f"🔍 **Interpretation:** This review has mixed or neutral sentiment with lower confidence.\n"
        
        return output
    
    def _format_sentiment_output_multiple(self, sentiment_data: Dict, review_count: int) -> str:
        """Format sentiment analysis output for multiple reviews"""
        if not sentiment_data:
            return "❌ No sentiment analysis results available."
        
        output = f"📊 **Sentiment Analysis Results** ({review_count} reviews)\n\n"
        
        # Overall sentiment
        overall_sentiment = sentiment_data.get('overall_sentiment', 'Unknown')
        confidence = sentiment_data.get('confidence', 0)
        output += f"**Overall Sentiment:** {overall_sentiment.title()} ({confidence:.1%} confidence)\n\n"
        
        # Distribution
        distribution = sentiment_data.get('distribution', {})
        if distribution:
            output += f"**Sentiment Distribution:**\n"
            for sentiment, percentage in distribution.items():
                output += f"• {sentiment.title()}: {percentage:.1%}\n"
            output += "\n"
        
        # Statistics
        individual_count = sentiment_data.get('individual_results', 0)
        if individual_count:
            output += f"**Key Statistics:**\n"
            output += f"• Reviews Analyzed: {individual_count}\n"
            output += f"• Average Confidence: {confidence:.1%}\n"
            output += f"• Analysis Type: {sentiment_data.get('type', 'unknown').title()}\n\n"
        
        # Insights
        output += "🔍 **Key Insights:**\n"
        if overall_sentiment.lower() == 'positive':
            output += "• Customers generally have a favorable opinion\n"
            output += "• Consider highlighting positive aspects in marketing\n"
        elif overall_sentiment.lower() == 'negative':
            output += "• There are concerns that need attention\n"
            output += "• Review negative feedback to identify improvement areas\n"
        else:
            output += "• Mixed or neutral feedback - dive deeper into specifics\n"
            output += "• Consider analyzing reviews by category or feature\n"
        
        return output
    
    def _format_summary_output(self, summary_data: Dict) -> str:
        """Format summary output"""
        if not summary_data:
            return "❌ No summary available."
        
        output = f"📝 **AI-Generated Summary**\n\n"
        
        # Main summary
        main_summary = summary_data.get('summary', '')
        if main_summary:
            output += f"**Key Findings:**\n{main_summary}\n\n"
        
        # Key themes (if available)
        themes = summary_data.get('key_themes', [])
        if themes:
            output += f"**Main Themes:**\n"
            for theme in themes[:5]:  # Show top 5 themes
                output += f"• {theme}\n"
            output += "\n"
        
        # Statistics
        stats = summary_data.get('statistics', {})
        if stats:
            output += f"**Summary Statistics:**\n"
            for key, value in stats.items():
                if isinstance(value, float):
                    output += f"• {key.replace('_', ' ').title()}: {value:.2f}\n"
                else:
                    output += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        return output
    
    def _create_sentiment_distribution_chart(self, sentiment_data: Dict) -> Optional[gr.Plot]:
        """Create a sentiment distribution chart"""
        try:
            distribution = sentiment_data.get('distribution', {})
            if not distribution:
                return None
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Convert percentages to actual labels and values
            sentiments = []
            counts = []
            for sentiment, percentage in distribution.items():
                sentiments.append(sentiment.title())
                counts.append(percentage * 100)  # Convert to percentage for display
            
            colors = ['#2E8B57', '#DC143C', '#FFD700']  # Green, Red, Gold
            
            bars = ax.bar(sentiments, counts, color=colors[:len(sentiments)])
            
            # Customize the plot
            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sentiment', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:.1f}%',
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
    
    def _create_single_review_chart(self, sentiment_data: Dict) -> Optional[gr.Plot]:
        """Create a sentiment scores chart for a single review"""
        try:
            scores = sentiment_data.get('scores', {})
            if not scores:
                return None
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sentiments = []
            values = []
            for sentiment, value in scores.items():
                sentiments.append(sentiment.title())
                values.append(value)
            
            colors = ['#2E8B57', '#DC143C']  # Green, Red
            
            bars = ax.bar(sentiments, values, color=colors[:len(sentiments)])
            
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
        """Create the Gradio interface"""
        
        # Custom CSS for better appearance
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .tab-nav {
            background-color: #f0f0f0;
        }
        """
        
        with gr.Blocks(css=css, title="Customer Sentiment Summarizer") as interface:
            
            # Header
            gr.Markdown("""
            <div class="main-header">
                <h1>🤖 Customer Sentiment Summarizer</h1>
                <p>Powered by BERT & BART AI | Advanced Dataset Analysis Platform</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Dataset Analysis
                with gr.TabItem("� Dataset Analysis"):
                    gr.Markdown("""
                    ### 🗄️ Kaggle Dataset Integration
                    Load and analyze customer review datasets directly from Kaggle for comprehensive sentiment analysis.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            kaggle_dataset_input = gr.Textbox(
                                label="� Kaggle Dataset ID",
                                placeholder="e.g., datafiniti/consumer-reviews-of-amazon-products",
                                value="datafiniti/consumer-reviews-of-amazon-products",
                                info="Enter the Kaggle dataset identifier (format: username/dataset-name)"
                            )
                        with gr.Column(scale=1):
                            kaggle_sample_size = gr.Slider(
                                label="📊 Sample Size",
                                minimum=10,
                                maximum=1000,
                                value=100,
                                step=10,
                                info="Number of reviews to analyze"
                            )
                    
                    with gr.Row():
                        kaggle_analyze_btn = gr.Button("� Analyze Dataset", variant="primary", size="lg")
                        dataset_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    gr.Markdown("""
                    **� Verified Working Datasets:**
                    - `datafiniti/consumer-reviews-of-amazon-products` - Amazon product reviews (28K+ reviews)
                    - `snap/amazon-fine-food-reviews` - Amazon fine food reviews 
                    - Search Kaggle for more review datasets using keywords like "amazon reviews", "product reviews", "customer feedback"
                    
                    **💡 Tips:**
                    - Start with smaller sample sizes (50-100) for quick testing
                    - Use larger samples (500+) for more comprehensive analysis
                    - The system automatically detects text and rating columns
                    """)
                    
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            dataset_sentiment_output = gr.Markdown(label="Sentiment Analysis")
                        with gr.Column():
                            dataset_summary_output = gr.Markdown(label="AI Summary & Insights")
                    
                    with gr.Row():
                        dataset_chart_output = gr.Plot(label="Sentiment Distribution")
                
                # Tab 2: Single Review Analysis
                with gr.TabItem("🔍 Single Review Analysis"):
                    gr.Markdown("""
                    ### Analyze Individual Reviews
                    Paste a single customer review to get detailed sentiment analysis and insights.
                    """)
                    
                    with gr.Row():
                        single_input = gr.Textbox(
                            label="📝 Customer Review",
                            placeholder="Paste a customer review here...",
                            lines=5
                        )
                    
                    with gr.Row():
                        single_analyze_btn = gr.Button("🔍 Analyze Review", variant="primary", size="lg")
                        single_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            single_sentiment_output = gr.Markdown(label="Sentiment Analysis")
                        with gr.Column():
                            single_summary_output = gr.Markdown(label="Key Insights")
                    
                    with gr.Row():
                        single_chart_output = gr.Plot(label="Sentiment Scores")
                
                # Tab 3: Multiple Reviews Analysis
                with gr.TabItem("📋 Multiple Reviews Analysis"):
                    gr.Markdown("""
                    ### Bulk Review Analysis
                    Paste multiple reviews (one per line or separated by blank lines) for comprehensive analysis.
                    """)
                    
                    multiple_input = gr.Textbox(
                        label="📝 Multiple Reviews",
                        placeholder="Paste multiple reviews here, one per line or separated by blank lines...",
                        lines=10
                    )
                    
                    with gr.Row():
                        multiple_analyze_btn = gr.Button("🔍 Analyze Reviews", variant="primary", size="lg")
                        multiple_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            multiple_sentiment_output = gr.Markdown(label="Sentiment Analysis")
                        with gr.Column():
                            multiple_summary_output = gr.Markdown(label="AI Summary")
                    
                    with gr.Row():
                        multiple_chart_output = gr.Plot(label="Sentiment Distribution")
                
                # Tab 4: Quick Guide
                with gr.TabItem("📖 Quick Guide"):
                    gr.Markdown("""
                    # 🚀 How to Use the Customer Sentiment Analysis Platform
                    
                    ## � Dataset Analysis Tab (Primary Feature)
                    - **Best for**: Large-scale analysis of thousands of customer reviews
                    - **How to use**: Enter a Kaggle dataset ID like `datafiniti/consumer-reviews-of-amazon-products`
                    - **Features**: Automated column detection, smart sampling, comprehensive analysis
                    - **Sample sizes**: Start with 50-100 for quick tests, use 500+ for detailed insights
                    
                    ### 🎯 Verified Working Datasets:
                    - `datafiniti/consumer-reviews-of-amazon-products` (28K+ Amazon reviews)
                    - `snap/amazon-fine-food-reviews` (Amazon food reviews)
                    - Search Kaggle for more using keywords: "amazon reviews", "product reviews", "customer feedback"
                    
                    ## 🔍 Single Review Analysis Tab
                    - **Best for**: Detailed analysis of individual reviews
                    - **How to use**: Copy and paste any customer review from Amazon, Google, etc.
                    - **Features**: Detailed sentiment scores and confidence levels
                    
                    ## 📋 Multiple Reviews Analysis Tab
                    - **Best for**: Comprehensive analysis of your own review collections
                    - **How to use**: Paste multiple reviews, one per line or separated by blank lines
                    - **Features**: Overall sentiment distribution, statistics, and AI summary
                    
                    ## 🤖 AI Models Used
                    - **Sentiment Analysis**: DistilBERT (fine-tuned for sentiment classification)
                    - **Text Summarization**: BART-large-CNN (for generating insights)
                    - **Performance**: Optimized for Apple Silicon (MPS) when available
                    
                    ## 💡 Tips for Best Results
                    - **Dataset Analysis**: Start small (50 samples) then scale up
                    - **Manual Analysis**: Include complete sentences for better accuracy
                    - **Multiple Reviews**: Provide varied reviews for balanced insights
                    - **Check Confidence**: Higher confidence scores indicate more reliable results
                    
                    ## � Technical Features
                    - **Smart Column Detection**: Automatically finds review text and ratings
                    - **Flexible Sampling**: Choose sample sizes based on your needs
                    - **Real-time Processing**: Live progress updates during analysis
                    - **Interactive Visualizations**: Charts showing sentiment distributions
                    - **Export-ready Results**: Copy results for reports or presentations
                    """)
                
                # Tab 5: Analysis History
                with gr.TabItem("📚 Analysis History"):
                    gr.Markdown("### Recent Analysis History")
                    history_refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
                    history_output = gr.Markdown("📊 Analysis history will appear here...")
            
            # Event handlers
            
            # Dataset analysis tab
            kaggle_analyze_btn.click(
                fn=self.analyze_kaggle_dataset,
                inputs=[kaggle_dataset_input, kaggle_sample_size],
                outputs=[dataset_sentiment_output, dataset_summary_output, dataset_chart_output]
            )
            
            dataset_clear_btn.click(
                fn=self.clear_outputs,
                outputs=[kaggle_dataset_input, dataset_sentiment_output, dataset_summary_output, dataset_chart_output]
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
            
            # History tab
            history_refresh_btn.click(
                fn=self.refresh_history,
                outputs=[history_output]
            )
        
        return interface

def main():
    """Main function to run the app"""
    try:
        # Create and launch the app
        app = SentimentApp()
        interface = app.create_interface()
        
        # Launch with appropriate settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Error starting the application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

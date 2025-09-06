"""
Gradio Frontend for Customer Sentiment Summarizer Chatbot
Provides an interactive interface for analyzing customer reviews
"""

import gradio as gr
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import time

# Add backend directory to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.append(backend_path)

try:
    from main_simple import CustomerSentimentSummarizer
    from bulk_import import BulkReviewImporter
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend not available: {e}")
    BACKEND_AVAILABLE = False

class SentimentChatbot:
    def __init__(self):
        """Initialize the Gradio chatbot interface"""
        self.analyzer = None
        self.bulk_importer = None
        self.chat_history = []
        self.analysis_history = []
        
        if BACKEND_AVAILABLE:
            try:
                print("🚀 Initializing AI models...")
                self.analyzer = CustomerSentimentSummarizer()
                self.bulk_importer = BulkReviewImporter()
                print("✅ AI models loaded successfully!")
            except Exception as e:
                print(f"❌ Error initializing AI models: {e}")
                self.analyzer = None
                self.bulk_importer = None
        
    def analyze_review(self, review_text: str, include_topics: bool = True, include_summary: bool = True) -> Tuple[str, str, str, object]:
        """
        Analyze a customer review and return formatted results
        
        Args:
            review_text: The customer review text
            include_topics: Whether to include topic analysis
            include_summary: Whether to include summary
            
        Returns:
            Tuple of (sentiment_result, topics_result, summary_result, chart)
        """
        if not self.analyzer:
            return (
                "❌ AI models not available. Please check your installation.",
                "❌ Topic analysis not available.",
                "❌ Summary not available.",
                None
            )
        
        if not review_text or not review_text.strip():
            return (
                "Please enter a customer review to analyze.",
                "No topics to extract.",
                "No summary to generate.",
                None
            )
        
        try:
            # Perform analysis
            result = self.analyzer.analyze_customer_feedback(
                review_text.strip(),
                include_topics=include_topics,
                include_summary=include_summary
            )
            
            # Store in history
            self.analysis_history.append({
                'timestamp': result.get('timestamp', ''),
                'input': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'sentiment': result.get('sentiment', {}).get('overall_sentiment', 'Unknown'),
                'processing_time': result.get('processing_time', 0)
            })
            
            # Format results
            sentiment_result = self._format_sentiment_output(result.get('sentiment', {}))
            topics_result = self._format_topics_output(result.get('topics', {}))
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Create sentiment chart
            chart = self._create_sentiment_chart(result.get('sentiment', {}))
            
            return sentiment_result, topics_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, error_msg, None
    
    def scrape_and_analyze(self, product_name: str, platform: str = "amazon", max_reviews: int = 30) -> Tuple[str, str, str, object]:
        """
        Scrape reviews from e-commerce sites and analyze them
        
        Args:
            product_name: Name of the product to search for
            platform: Platform to scrape from
            max_reviews: Maximum number of reviews to scrape
            
        Returns:
            Tuple of (scraping_result, sentiment_result, summary_result, chart)
        """
        if not self.analyzer:
            return (
                "❌ AI models not available. Please check your installation.",
                "❌ Analysis not available.",
                "❌ Summary not available.",
                None
            )
        
        if not product_name or not product_name.strip():
            return (
                "Please enter a product name to search for reviews.",
                "No analysis to perform.",
                "No summary to generate.",
                None
            )
        
        try:
            # Scrape and analyze reviews
            result = self.analyzer.scrape_and_analyze_reviews(
                product_name.strip(),
                platform=platform,
                max_reviews=max_reviews
            )
            
            if not result.get('success'):
                error_msg = f"❌ Scraping failed: {result.get('error', 'Unknown error')}"
                return error_msg, error_msg, error_msg, None
            
            # Store in history
            self.analysis_history.append({
                'timestamp': result.get('analysis', {}).get('timestamp', ''),
                'input': f"Scraped: {product_name} ({platform})",
                'sentiment': result.get('analysis', {}).get('sentiment', {}).get('overall_sentiment', 'Unknown'),
                'processing_time': result.get('total_time', 0)
            })
            
            # Format results
            scraping_result = self._format_scraping_output(result)
            
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
        # Check if we have a user-friendly message
        if result.get('user_message'):
            return result['user_message']
        
        if not result.get('success'):
            output = f"❌ **Scraping Failed**\n\n"
            
            errors = result.get('errors', [])
            if errors:
                output += "**Errors encountered:**\n"
                for error in errors:
                    output += f"• {error}\n"
                output += "\n"
            
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
        
        # Show sources info if multiple
        scraping_details = result.get('scraping_details', {})
        if scraping_details.get('sources_attempted'):
            output += f"**Sources Attempted:** {', '.join(scraping_details['sources_attempted'])}\n"
            output += f"**Sources Successful:** {', '.join(scraping_details['sources_successful'])}\n\n"
            
            # Show any errors from sources that failed
            if scraping_details.get('errors'):
                output += "**Partial Errors:**\n"
                for error in scraping_details['errors']:
                    output += f"• {error}\n"
                output += "\n"
        
        # Show sample reviews
        sample_reviews = result.get('sample_reviews', [])
        if sample_reviews:
            output += "**Sample Reviews:**\n"
            for i, review in enumerate(sample_reviews[:2], 1):
                review_preview = review[:150] + "..." if len(review) > 150 else review
                output += f"{i}. {review_preview}\n\n"
        
        return output
    
    def analyze_file_upload(self, file_obj, text_column: str = "review_text", separator: str = "\\n\\n") -> Tuple[str, str, str, object]:
        """
        Analyze reviews from uploaded file
        
        Args:
            file_obj: Uploaded file object from Gradio
            text_column: Column name for CSV/Excel files
            separator: Separator for TXT files
            
        Returns:
            Tuple of (sentiment_result, topics_result, summary_result, chart)
        """
        if not self.analyzer or not self.bulk_importer:
            return (
                "❌ AI models not available. Please check your installation.",
                "❌ Topic analysis not available.",
                "❌ Summary not available.",
                None
            )
        
        if not file_obj:
            return (
                "Please upload a file containing reviews.",
                "No topics to extract.",
                "No summary to generate.",
                None
            )
        
        try:
            # Import reviews from uploaded file
            file_path = file_obj.name
            
            # Determine import parameters based on file type
            if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                reviews = self.bulk_importer.import_reviews(file_path, text_column=text_column)
            elif file_path.lower().endswith('.json'):
                reviews = self.bulk_importer.import_reviews(file_path, text_field='text')
            else:  # TXT or other
                # Convert \\n\\n to actual newlines
                actual_separator = separator.replace('\\n', '\n')
                reviews = self.bulk_importer.import_reviews(file_path, separator=actual_separator)
            
            if not reviews:
                return (
                    "❌ No valid reviews found in the uploaded file.",
                    "No topics to extract.",
                    "No summary to generate.",
                    None
                )
            
            # Perform analysis
            result = self.analyzer.analyze_customer_feedback(reviews)
            
            # Store in history
            self.analysis_history.append({
                'timestamp': result.get('timestamp', ''),
                'input': f"File upload: {len(reviews)} reviews",
                'sentiment': result.get('sentiment', {}).get('overall_sentiment', 'Unknown'),
                'processing_time': result.get('processing_time', 0)
            })
            
            # Format results
            sentiment_result = self._format_sentiment_output_multiple(result.get('sentiment', {}), len(reviews))
            topics_result = self._format_topics_output(result.get('topics', {}))
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(result.get('sentiment', {}))
            
            return sentiment_result, topics_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error processing file: {str(e)}"
            return error_msg, error_msg, error_msg, None
    
    def analyze_multiple_reviews(self, reviews_text: str) -> Tuple[str, str, str, object]:
        """
        Analyze multiple customer reviews (one per line)
        
        Args:
            reviews_text: Multiple reviews separated by newlines
            
        Returns:
            Tuple of (sentiment_result, topics_result, summary_result, chart)
        """
        if not self.analyzer:
            return (
                "❌ AI models not available. Please check your installation.",
                "❌ Topic analysis not available.",
                "❌ Summary not available.",
                None
            )
        
        if not reviews_text or not reviews_text.strip():
            return (
                "Please enter customer reviews to analyze (one per line).",
                "No topics to extract.",
                "No summary to generate.",
                None
            )
        
        try:
            # Split reviews by newlines and filter empty lines
            reviews = [review.strip() for review in reviews_text.split('\n') if review.strip()]
            
            if not reviews:
                return (
                    "No valid reviews found. Please enter at least one review.",
                    "No topics to extract.",
                    "No summary to generate.",
                    None
                )
            
            # Perform analysis
            result = self.analyzer.analyze_customer_feedback(reviews)
            
            # Store in history
            self.analysis_history.append({
                'timestamp': result.get('timestamp', ''),
                'input': f"{len(reviews)} reviews",
                'sentiment': result.get('sentiment', {}).get('overall_sentiment', 'Unknown'),
                'processing_time': result.get('processing_time', 0)
            })
            
            # Format results
            sentiment_result = self._format_sentiment_output_multiple(result.get('sentiment', {}), len(reviews))
            topics_result = self._format_topics_output(result.get('topics', {}))
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(result.get('sentiment', {}))
            
            return sentiment_result, topics_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, error_msg, None
        """
        Analyze multiple customer reviews (one per line)
        
        Args:
            reviews_text: Multiple reviews separated by newlines
            
        Returns:
            Tuple of (sentiment_result, topics_result, summary_result, chart)
        """
        if not self.analyzer:
            return (
                "❌ AI models not available. Please check your installation.",
                "❌ Topic analysis not available.",
                "❌ Summary not available.",
                None
            )
        
        if not reviews_text or not reviews_text.strip():
            return (
                "Please enter customer reviews to analyze (one per line).",
                "No topics to extract.",
                "No summary to generate.",
                None
            )
        
        try:
            # Split reviews by newlines and filter empty lines
            reviews = [review.strip() for review in reviews_text.split('\n') if review.strip()]
            
            if not reviews:
                return (
                    "No valid reviews found. Please enter at least one review.",
                    "No topics to extract.",
                    "No summary to generate.",
                    None
                )
            
            # Perform analysis
            result = self.analyzer.analyze_customer_feedback(reviews)
            
            # Store in history
            self.analysis_history.append({
                'timestamp': result.get('timestamp', ''),
                'input': f"{len(reviews)} reviews",
                'sentiment': result.get('sentiment', {}).get('overall_sentiment', 'Unknown'),
                'processing_time': result.get('processing_time', 0)
            })
            
            # Format results
            sentiment_result = self._format_sentiment_output_multiple(result.get('sentiment', {}), len(reviews))
            topics_result = self._format_topics_output(result.get('topics', {}))
            summary_result = self._format_summary_output(result.get('summary', {}))
            
            # Create sentiment distribution chart
            chart = self._create_sentiment_distribution_chart(result.get('sentiment', {}))
            
            return sentiment_result, topics_result, summary_result, chart
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, error_msg, None
    
    def _format_sentiment_output(self, sentiment_data: Dict) -> str:
        """Format sentiment analysis output for single review"""
        if not sentiment_data:
            return "❌ No sentiment data available."
        
        sentiment = sentiment_data.get('overall_sentiment', 'Unknown')
        confidence = sentiment_data.get('confidence', 0.0)
        scores = sentiment_data.get('scores', {})
        
        output = f"🎯 **Sentiment Analysis**\n\n"
        output += f"**Overall Sentiment:** {sentiment}\n"
        output += f"**Confidence:** {confidence:.1%}\n\n"
        
        if scores:
            output += "**Detailed Scores:**\n"
            for label, score in scores.items():
                emoji = "😊" if label.lower() == "positive" else "😞" if label.lower() == "negative" else "😐"
                output += f"  {emoji} {label.title()}: {score:.1%}\n"
        
        # Add confidence interpretation
        if confidence > 0.8:
            output += "\n✅ **High confidence** in sentiment prediction."
        elif confidence > 0.6:
            output += "\n⚠️ **Medium confidence** in sentiment prediction."
        else:
            output += "\n❓ **Low confidence** in sentiment prediction."
        
        return output
    
    def _format_sentiment_output_multiple(self, sentiment_data: Dict, review_count: int) -> str:
        """Format sentiment analysis output for multiple reviews"""
        if not sentiment_data:
            return "❌ No sentiment data available."
        
        sentiment = sentiment_data.get('overall_sentiment', 'Unknown')
        confidence = sentiment_data.get('confidence', 0.0)
        distribution = sentiment_data.get('distribution', {})
        
        output = f"🎯 **Sentiment Analysis ({review_count} reviews)**\n\n"
        output += f"**Overall Sentiment:** {sentiment}\n"
        output += f"**Confidence:** {confidence:.1%}\n\n"
        
        if distribution:
            output += "**Sentiment Distribution:**\n"
            for label, percentage in distribution.items():
                emoji = "😊" if label.lower() == "positive" else "😞" if label.lower() == "negative" else "😐"
                output += f"  {emoji} {label.title()}: {percentage:.1%}\n"
        
        return output
    
    def _format_topics_output(self, topics_data: Dict) -> str:
        """Format topic analysis output"""
        if not topics_data or 'error' in topics_data:
            return f"❌ Topic analysis failed: {topics_data.get('error', 'Unknown error')}"
        
        topics = topics_data.get('topics', [])
        method = topics_data.get('method', 'unknown')
        
        if not topics:
            return "🔍 **Topic Analysis**\n\nNo significant topics detected in the review(s)."
        
        output = f"🔍 **Topic Analysis** (using {method.upper()})\n\n"
        output += f"**Detected Topics ({len(topics)}):**\n\n"
        
        for i, topic in enumerate(topics, 1):
            label = topic.get('label', f'Topic {i}')
            keywords = topic.get('keywords', [])
            
            output += f"**{i}. {label}**\n"
            if keywords:
                output += f"   Keywords: {', '.join(keywords[:5])}\n"
            output += "\n"
        
        return output
    
    def _format_summary_output(self, summary_data: Dict) -> str:
        """Format summary output"""
        if not summary_data or not summary_data.get('success', False):
            return "❌ Summary generation failed."
        
        summary_text = summary_data.get('text', 'No summary available.')
        original_length = summary_data.get('original_length', 0)
        summary_length = summary_data.get('summary_length', 0)
        compression_ratio = summary_data.get('compression_ratio', 0)
        
        output = f"📝 **Summary**\n\n"
        output += f"{summary_text}\n\n"
        output += f"**Summary Statistics:**\n"
        output += f"  • Original: {original_length} words\n"
        output += f"  • Summary: {summary_length} words\n"
        output += f"  • Compression: {compression_ratio:.1%}\n"
        
        return output
    
    def _create_sentiment_chart(self, sentiment_data: Dict) -> object:
        """Create a sentiment scores chart"""
        try:
            scores = sentiment_data.get('scores', {})
            if not scores:
                return None
            
            plt.figure(figsize=(8, 4))
            labels = list(scores.keys())
            values = list(scores.values())
            colors = ['#28a745' if 'pos' in label.lower() else '#dc3545' if 'neg' in label.lower() else '#6c757d' 
                     for label in labels]
            
            bars = plt.bar(labels, values, color=colors, alpha=0.7)
            plt.title('Sentiment Scores', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
    
    def _create_sentiment_distribution_chart(self, sentiment_data: Dict) -> object:
        """Create a sentiment distribution pie chart"""
        try:
            distribution = sentiment_data.get('distribution', {})
            if not distribution:
                return None
            
            plt.figure(figsize=(8, 6))
            labels = list(distribution.keys())
            values = list(distribution.values())
            colors = ['#28a745', '#dc3545', '#6c757d']  # Green, Red, Gray
            
            # Filter out zero values
            filtered_data = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 0.01]
            if not filtered_data:
                return None
            
            labels, values, colors = zip(*filtered_data)
            
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating distribution chart: {e}")
            return None
    
    def get_analysis_history(self) -> str:
        """Get formatted analysis history"""
        if not self.analysis_history:
            return "No analysis history available."
        
        output = "📊 **Analysis History**\n\n"
        for i, entry in enumerate(reversed(self.analysis_history[-10:]), 1):  # Show last 10
            output += f"**{i}.** {entry['timestamp']}\n"
            output += f"   Input: {entry['input']}\n"
            output += f"   Sentiment: {entry['sentiment']}\n"
            output += f"   Processing Time: {entry['processing_time']}s\n\n"
        
        return output

def create_interface():
    """Create and configure the Gradio interface"""
    chatbot = SentimentChatbot()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #007bff, #0056b3);
        border: none;
    }
    .gr-button-primary:hover {
        background: linear-gradient(45deg, #0056b3, #003d82);
    }
    """
    
    with gr.Blocks(css=css, title="Customer Sentiment Summarizer", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2c3e50; margin-bottom: 10px;">🎯 Customer Sentiment Summarizer Chatbot</h1>
            <p style="color: #7f8c8d; font-size: 16px;">
                Analyze customer reviews with AI-powered sentiment analysis, topic extraction, and summarization
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Web Scraping Tab
            with gr.TabItem("🕷️ Auto-Scrape Reviews"):
                with gr.Row():
                    with gr.Column(scale=3):
                        product_input = gr.Textbox(
                            label="Product Name",
                            placeholder="Enter product name (e.g., 'iPhone 15', 'Samsung Galaxy S24')",
                            lines=1
                        )
                        
                        with gr.Row():
                            platform_dropdown = gr.Dropdown(
                                choices=["amazon", "flipkart", "multiple"],
                                value="amazon",
                                label="E-commerce Platform"
                            )
                            max_reviews_slider = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=10,
                                label="Max Reviews"
                            )
                        
                        with gr.Row():
                            scrape_btn = gr.Button("🕷️ Scrape & Analyze", variant="primary", size="lg")
                            scrape_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding: 20px; background: #e8f5e8; border-radius: 10px;">
                            <h3>🕷️ Auto-Scraping Features:</h3>
                            <ul>
                                <li>🔍 Search products automatically</li>
                                <li>📊 Scrape reviews from major platforms</li>
                                <li>🧠 Analyze sentiment & topics</li>
                                <li>📝 Generate summaries</li>
                            </ul>
                            <br>
                            <p><strong>Supported Platforms:</strong></p>
                            <ul>
                                <li>🛒 Amazon</li>
                                <li>🛍️ Flipkart</li>
                                <li>🌐 Multiple sources</li>
                            </ul>
                            <br>
                            <p><strong>Example:</strong><br>
                            Product: "iPhone 15 Pro"<br>
                            Platform: Amazon<br>
                            Max Reviews: 50</p>
                            <br>
                            <p style="background: #fff3cd; padding: 10px; border-radius: 5px; font-size: 12px;">
                            <strong>💡 Note:</strong> If auto-scraping is blocked, use the manual tabs below to copy/paste reviews directly!
                            </p>
                        </div>
                        """)
                
                with gr.Row():
                    with gr.Column():
                        scrape_output = gr.Markdown(label="Scraping Results")
                    with gr.Column():
                        scrape_sentiment_output = gr.Markdown(label="Sentiment Analysis")
                
                with gr.Row():
                    with gr.Column():
                        scrape_summary_output = gr.Markdown(label="Summary")
                    with gr.Column():
                        scrape_chart_output = gr.Plot(label="Sentiment Distribution")
            
            # Single Review Analysis Tab
            with gr.TabItem("📝 Single Review Analysis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        single_input = gr.Textbox(
                            label="Customer Review",
                            placeholder="Enter a customer review here...",
                            lines=5,
                            max_lines=10
                        )
                        
                        with gr.Row():
                            single_analyze_btn = gr.Button("🔍 Analyze Review", variant="primary", size="lg")
                            single_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                            <h3>💡 How to use:</h3>
                            <ol>
                                <li>Paste a customer review in the text box</li>
                                <li>Click "Analyze Review"</li>
                                <li>View sentiment, topics, and summary</li>
                            </ol>
                            <br>
                            <p><strong>Example iPhone 15 Pro Max review:</strong><br>
                            <em>"The iPhone 15 Pro Max is absolutely amazing! The camera quality is outstanding, especially the new 5x optical zoom. Battery life easily lasts all day, even with heavy usage. The titanium build feels premium and lightweight. Face ID is lightning fast. Only complaint is the price, but the quality justifies it. Highly recommend!"</em></p>
                            <br>
                            <p style="background: #d1ecf1; padding: 8px; border-radius: 4px; font-size: 12px;">
                            <strong>💡 Tip:</strong> This works great when auto-scraping is blocked!
                            </p>
                        </div>
                        """)
                
                with gr.Row():
                    with gr.Column():
                        single_sentiment_output = gr.Markdown(label="Sentiment Analysis")
                    with gr.Column():
                        single_topics_output = gr.Markdown(label="Topic Analysis")
                
                with gr.Row():
                    with gr.Column():
                        single_summary_output = gr.Markdown(label="Summary")
                    with gr.Column():
                        single_chart_output = gr.Plot(label="Sentiment Scores")
            
            # Multiple Reviews Analysis Tab
            with gr.TabItem("📊 Multiple Reviews Analysis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.TabItem("📝 Manual Input"):
                                multiple_input = gr.Textbox(
                                    label="Customer Reviews (one per line)",
                                    placeholder="Enter multiple customer reviews, one per line...",
                                    lines=8,
                                    max_lines=15
                                )
                            
                            with gr.TabItem("📁 File Upload"):
                                file_upload = gr.File(
                                    label="Upload Reviews File",
                                    file_types=[".txt", ".csv", ".json", ".xlsx"]
                                )
                                file_text_column = gr.Textbox(
                                    label="Text Column Name (for CSV/Excel)",
                                    placeholder="review_text",
                                    value="review_text"
                                )
                                file_separator = gr.Textbox(
                                    label="Separator (for TXT files)",
                                    placeholder="\\n\\n",
                                    value="\\n\\n"
                                )
                        
                        with gr.Row():
                            multiple_analyze_btn = gr.Button("🔍 Analyze Reviews", variant="primary", size="lg")
                            file_analyze_btn = gr.Button("📁 Analyze File", variant="primary", size="lg") 
                            multiple_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                            <h3>💡 How to use:</h3>
                            <ol>
                                <li>Enter multiple reviews, one per line</li>
                                <li>Click "Analyze Reviews"</li>
                                <li>View aggregated analysis</li>
                            </ol>
                            <br>
                            <p><strong>Manual Entry:</strong><br>
                            <em>Amazing camera quality, love the 5x zoom feature!<br>
                            Battery life is excellent, lasts all day easily<br>
                            Titanium feels premium but phone is still heavy<br>
                            Price is too high for what you get<br>
                            Face ID works perfectly even in low light<br>
                            Love the new Action Button customization</em></p>
                            <br>
                            <p><strong>File Upload:</strong><br>
                            Upload .txt, .csv, .json, or .xlsx files containing reviews<br>
                            <em>Example formats supported:</em><br>
                            • TXT: One review per paragraph<br>
                            • CSV: Column with review text<br>
                            • JSON: Array of review objects<br>
                            • Excel: Spreadsheet with review column</p>
                            <br>
                            <p style="background: #d1ecf1; padding: 8px; border-radius: 4px; font-size: 12px;">
                            <strong>💡 Tip:</strong> File upload is perfect for bulk analysis!
                            </p>
                        </div>
                        """)
                
                with gr.Row():
                    with gr.Column():
                        multiple_sentiment_output = gr.Markdown(label="Overall Sentiment Analysis")
                    with gr.Column():
                        multiple_topics_output = gr.Markdown(label="Topic Analysis")
                
                with gr.Row():
                    with gr.Column():
                        multiple_summary_output = gr.Markdown(label="Summary")
                    with gr.Column():
                        multiple_chart_output = gr.Plot(label="Sentiment Distribution")
            
            # Analysis History Tab
            with gr.TabItem("📈 Analysis History"):
                with gr.Row():
                    history_output = gr.Markdown(label="Recent Analysis History")
                    history_refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
        
        # Event handlers
        scrape_btn.click(
            fn=chatbot.scrape_and_analyze,
            inputs=[product_input, platform_dropdown, max_reviews_slider],
            outputs=[scrape_output, scrape_sentiment_output, scrape_summary_output, scrape_chart_output]
        )
        
        scrape_clear_btn.click(
            fn=lambda: ("", "amazon", 30, "", "", "", None),
            outputs=[product_input, platform_dropdown, max_reviews_slider, scrape_output, scrape_sentiment_output, scrape_summary_output, scrape_chart_output]
        )
        
        single_analyze_btn.click(
            fn=chatbot.analyze_review,
            inputs=[single_input],
            outputs=[single_sentiment_output, single_topics_output, single_summary_output, single_chart_output]
        )
        
        single_clear_btn.click(
            fn=lambda: ("", "", "", "", None),
            outputs=[single_input, single_sentiment_output, single_topics_output, single_summary_output, single_chart_output]
        )
        
        multiple_analyze_btn.click(
            fn=chatbot.analyze_multiple_reviews,
            inputs=[multiple_input],
            outputs=[multiple_sentiment_output, multiple_topics_output, multiple_summary_output, multiple_chart_output]
        )
        
        file_analyze_btn.click(
            fn=chatbot.analyze_file_upload,
            inputs=[file_upload, file_text_column, file_separator],
            outputs=[multiple_sentiment_output, multiple_topics_output, multiple_summary_output, multiple_chart_output]
        )
        
        multiple_clear_btn.click(
            fn=lambda: ("", None, "", "", "", "", None),
            outputs=[multiple_input, file_upload, file_text_column, file_separator, multiple_sentiment_output, multiple_topics_output, multiple_summary_output, multiple_chart_output]
        )
        
        history_refresh_btn.click(
            fn=chatbot.get_analysis_history,
            outputs=[history_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid #e9ecef;">
            <p style="color: #6c757d;">
                Powered by 🤗 Transformers • Built with ❤️ using Gradio
            </p>
        </div>
        """)
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Starting Customer Sentiment Summarizer Chatbot...")
    
    if not BACKEND_AVAILABLE:
        print("⚠️ Warning: Backend models not available. Please install requirements first.")
        print("Run: pip install -r requirements.txt")
    
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default port for Hugging Face Spaces
        share=False,            # Set to True to create public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()

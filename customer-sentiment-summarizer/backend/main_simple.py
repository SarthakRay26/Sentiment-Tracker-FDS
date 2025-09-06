"""
Simplified Main Backend Integration Module
Focuses on sentiment analysis with optional topic and summary components
"""

import os
import sys
from typing import Dict, List, Union, Optional
import json
import time

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Always import sentiment analyzer (core functionality)
from sentiment import SentimentAnalyzer

# Try to import scraper
try:
    from scraper import ReviewScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Review scraper not available: {e}")
    SCRAPER_AVAILABLE = False

# Try to import database
try:
    from database import ReviewDatabase
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Database not available: {e}")
    DATABASE_AVAILABLE = False

class CustomerSentimentSummarizer:
    def __init__(self, 
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 summarization_model: str = "facebook/bart-large-cnn",
                 use_bertopic: bool = False,  # Disabled by default due to compatibility
                 num_topics: int = 5):
        """
        Initialize the customer sentiment summarizer system
        
        Args:
            sentiment_model: HuggingFace model for sentiment analysis
            summarization_model: HuggingFace model for text summarization
            use_bertopic: Whether to use BERTopic for topic modeling (disabled by default)
            num_topics: Number of topics to extract
        """
        self.sentiment_analyzer = None
        self.topic_extractor = None
        self.text_summarizer = None
        self.review_scraper = None
        self.database = None
        
        print("🚀 Initializing Customer Sentiment Summarizer...")
        
        # Initialize core sentiment analysis (always available)
        print("📊 Loading sentiment analysis model...")
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        
        # Initialize review scraper (if available)
        if SCRAPER_AVAILABLE:
            print("🕷️ Initializing review scraper...")
            self.review_scraper = ReviewScraper()
            print("✅ Review scraper ready")
        else:
            print("🕷️ Review scraper not available")
        
        # Initialize database (if available)
        if DATABASE_AVAILABLE:
            print("🗄️ Initializing database...")
            self.database = ReviewDatabase("sqlite", "data/reviews.db")
            print("✅ Database ready")
        else:
            print("🗄️ Database not available")
        
        # Try to initialize optional components
        self._initialize_optional_components(summarization_model, use_bertopic, num_topics)
        
        print("✅ Customer Sentiment Summarizer ready!")
    
    def _initialize_optional_components(self, summarization_model, use_bertopic, num_topics):
        """Initialize optional components with graceful fallback"""
        
        # Try to initialize topic extractor
        if use_bertopic:
            try:
                from topics import TopicExtractor
                print("🔍 Loading topic extraction model...")
                self.topic_extractor = TopicExtractor(use_bertopic=use_bertopic, num_topics=num_topics)
                print("✅ Topic extractor loaded")
            except Exception as e:
                print(f"⚠️ Topic extractor failed to load: {e}")
                print("Continuing without topic extraction...")
        else:
            print("🔍 Topic extraction disabled")
        
        # Try to initialize summarizer
        try:
            from summarizer import TextSummarizer
            print("📝 Loading text summarization model...")
            self.text_summarizer = TextSummarizer(summarization_model)
            print("✅ Text summarizer loaded")
        except Exception as e:
            print(f"⚠️ Text summarizer failed to load: {e}")
            print("Continuing without summarization...")
    
    def analyze_customer_feedback(self, 
                                text_input: Union[str, List[str]], 
                                include_topics: bool = True,
                                include_summary: bool = True,
                                max_summary_length: int = 150) -> Dict:
        """
        Complete analysis of customer feedback including sentiment, topics, and summary
        
        Args:
            text_input: Single review string or list of review strings
            include_topics: Whether to extract topics (only if topic extractor available)
            include_summary: Whether to generate summary (only if summarizer available)
            max_summary_length: Maximum length of summary
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        
        # Convert input to list for consistent processing
        if isinstance(text_input, str):
            texts = [text_input]
            single_input = True
        else:
            texts = text_input
            single_input = False
        
        if not texts or all(not text.strip() for text in texts):
            return {
                'error': 'No valid text provided',
                'success': False,
                'processing_time': 0
            }
        
        results = {
            'input_type': 'single' if single_input else 'multiple',
            'text_count': len(texts),
            'success': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': 0,
            'components_used': []
        }
        
        try:
            # 1. Sentiment Analysis (always available)
            print("📊 Analyzing sentiment...")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(texts)
            results['sentiment'] = self._format_sentiment_results(sentiment_result, single_input)
            results['components_used'].append('sentiment')
            
            # 2. Topic Extraction (if available and requested)
            if include_topics and self.topic_extractor:
                print("🔍 Extracting topics...")
                try:
                    topic_result = self.topic_extractor.extract_topics(texts)
                    results['topics'] = self._format_topic_results(topic_result)
                    results['components_used'].append('topics')
                except Exception as e:
                    print(f"⚠️ Topic extraction failed: {e}")
                    results['topics'] = {'error': str(e), 'topics': []}
            elif include_topics:
                results['topics'] = {'message': 'Topic extractor not available', 'topics': []}
            
            # 3. Text Summarization (if available and requested)
            if include_summary and self.text_summarizer:
                print("📝 Generating summary...")
                try:
                    if single_input:
                        summary_result = self.text_summarizer.summarize_text(texts[0], max_length=max_summary_length)
                    else:
                        summary_result = self.text_summarizer.summarize_multiple_reviews(texts, max_length=max_summary_length)
                    results['summary'] = self._format_summary_results(summary_result)
                    results['components_used'].append('summary')
                except Exception as e:
                    print(f"⚠️ Summarization failed: {e}")
                    results['summary'] = {'error': str(e), 'text': 'Summary not available', 'success': False}
            elif include_summary:
                results['summary'] = {'message': 'Text summarizer not available', 'text': self._create_simple_summary(texts), 'success': True}
            
            # Calculate processing time
            results['processing_time'] = round(time.time() - start_time, 2)
            
            # Add insights
            results['insights'] = self._generate_insights(results)
            
            print(f"✅ Analysis complete in {results['processing_time']} seconds")
            print(f"Components used: {', '.join(results['components_used'])}")
            return results
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            results['error'] = str(e)
            results['success'] = False
            results['processing_time'] = round(time.time() - start_time, 2)
            return results
    
    def _format_sentiment_results(self, sentiment_result: Dict, single_input: bool) -> Dict:
        """Format sentiment analysis results for output"""
        if single_input:
            return {
                'overall_sentiment': sentiment_result.get('sentiment', 'Neutral'),
                'confidence': round(sentiment_result.get('confidence', 0.0), 3),
                'scores': {k: round(v, 3) for k, v in sentiment_result.get('scores', {}).items()},
                'type': 'single'
            }
        else:
            overall = sentiment_result.get('overall_sentiment', {})
            return {
                'overall_sentiment': overall.get('sentiment', 'Neutral'),
                'confidence': round(overall.get('confidence', 0.0), 3),
                'distribution': {k: round(v, 3) for k, v in overall.get('distribution', {}).items()},
                'individual_results': len(sentiment_result.get('individual_results', [])),
                'type': 'multiple'
            }
    
    def _format_topic_results(self, topic_result: Dict) -> Dict:
        """Format topic extraction results for output"""
        if 'error' in topic_result:
            return {
                'error': topic_result['error'],
                'topics': [],
                'method': topic_result.get('method', 'unknown')
            }
        
        return {
            'topics': [
                {
                    'label': topic['label'],
                    'id': topic['id'],
                    'keywords': [word[0] if isinstance(word, tuple) else word 
                               for word in topic.get('words', [])[:3]]
                }
                for topic in topic_result.get('topics', [])
            ],
            'method': topic_result.get('method', 'unknown'),
            'count': len(topic_result.get('topics', []))
        }
    
    def _format_summary_results(self, summary_result: Dict) -> Dict:
        """Format summarization results for output"""
        return {
            'text': summary_result.get('summary', 'No summary available'),
            'original_length': summary_result.get('original_length', 0),
            'summary_length': summary_result.get('summary_length', 0),
            'compression_ratio': round(summary_result.get('compression_ratio', 0), 3),
            'model_used': summary_result.get('model_used', 'unknown'),
            'success': summary_result.get('success', False)
        }
    
    def _create_simple_summary(self, texts: List[str]) -> str:
        """Create a simple extractive summary as fallback"""
        if not texts:
            return "No text to summarize."
        
        # Combine all texts
        combined = " ".join(texts)
        
        # Extract key sentences (simple approach)
        sentences = [s.strip() for s in combined.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return "Unable to create summary from provided text."
        
        # Take first and most informative sentences
        if len(sentences) == 1:
            return sentences[0] + "."
        elif len(sentences) <= 3:
            return ". ".join(sentences) + "."
        else:
            # Take first sentence and one with positive/negative keywords
            summary_parts = [sentences[0]]
            
            keywords = ['good', 'bad', 'excellent', 'terrible', 'love', 'hate', 'recommend', 'disappointed']
            for sentence in sentences[1:]:
                if any(keyword in sentence.lower() for keyword in keywords):
                    summary_parts.append(sentence)
                    break
            
            if len(summary_parts) == 1 and len(sentences) > 1:
                summary_parts.append(sentences[-1])
            
            return ". ".join(summary_parts) + "."
    
    def _generate_insights(self, results: Dict) -> Dict:
        """Generate additional insights from the analysis"""
        insights = {}
        
        # Sentiment insights
        if 'sentiment' in results:
            sentiment = results['sentiment']['overall_sentiment']
            confidence = results['sentiment']['confidence']
            
            if confidence > 0.8:
                insights['sentiment_confidence'] = 'High'
            elif confidence > 0.6:
                insights['sentiment_confidence'] = 'Medium'
            else:
                insights['sentiment_confidence'] = 'Low'
            
            insights['sentiment_summary'] = f"Overall sentiment is {sentiment.lower()} with {insights['sentiment_confidence'].lower()} confidence."
        
        # Topic insights
        if 'topics' in results and results['topics'].get('count', 0) > 0:
            topic_count = results['topics']['count']
            insights['topic_diversity'] = 'High' if topic_count >= 3 else 'Low'
            
            main_topics = [topic['label'] for topic in results['topics']['topics'][:2]]
            if main_topics:
                insights['main_topics'] = main_topics
                insights['topic_summary'] = f"Main discussion topics: {', '.join(main_topics)}"
        
        # Summary insights
        if 'summary' in results and results['summary']['success']:
            compression = results['summary'].get('compression_ratio', 0)
            if compression < 0.3:
                insights['summary_quality'] = 'Highly compressed'
            elif compression < 0.6:
                insights['summary_quality'] = 'Well compressed'
            else:
                insights['summary_quality'] = 'Lightly compressed'
        
        # Component availability
        insights['available_features'] = results.get('components_used', [])
        
        return insights
    
    def scrape_and_analyze_reviews(self, 
                                  product_name: str, 
                                  platform: str = "amazon",
                                  max_reviews: int = 50,
                                  include_topics: bool = True,
                                  include_summary: bool = True,
                                  use_database: bool = True) -> Dict:
        """
        Scrape reviews from e-commerce platforms and analyze them
        
        Args:
            product_name: Name of the product to search for
            platform: Platform to scrape from ("amazon", "flipkart", "multiple")
            max_reviews: Maximum number of reviews to scrape
            include_topics: Whether to extract topics
            include_summary: Whether to generate summary
            use_database: Whether to store results in database
            
        Returns:
            Dictionary containing scraping results and analysis
        """
        if not self.review_scraper:
            return {
                'error': 'Review scraper not available. Please install required packages.',
                'success': False
            }
        
        print(f"🕷️ Starting to scrape reviews for: {product_name}")
        start_time = time.time()
        
        try:
            # Store product in database if available
            product_id = None
            if use_database and self.database:
                try:
                    product_id = self.database.store_product(product_name, product_name, "general")
                    print(f"💾 Stored product in database with ID: {product_id}")
                except Exception as e:
                    print(f"⚠️ Failed to store product in database: {e}")
            
            # Scrape reviews based on platform
            if platform.lower() == "amazon":
                scraping_result = self.review_scraper.scrape_amazon_reviews(product_name, max_reviews)
            elif platform.lower() == "flipkart":
                scraping_result = self.review_scraper.scrape_flipkart_reviews(product_name, max_reviews)
            elif platform.lower() == "multiple":
                scraping_result = self.review_scraper.scrape_multiple_sources(product_name, max_reviews//2)
            else:
                return {
                    'error': f'Unsupported platform: {platform}. Use "amazon", "flipkart", or "multiple"',
                    'success': False
                }
            
            if not scraping_result.get('success') or not scraping_result.get('reviews'):
                return {
                    'error': scraping_result.get('error', 'No reviews found'),
                    'scraping_result': scraping_result,
                    'success': False,
                    'scraping_time': round(time.time() - start_time, 2)
                }
            
            # Extract review texts for analysis
            review_texts = self.review_scraper.get_review_texts_only(scraping_result)
            
            if not review_texts:
                return {
                    'error': 'No review texts found to analyze',
                    'scraping_result': scraping_result,
                    'success': False,
                    'scraping_time': round(time.time() - start_time, 2)
                }
            
            scraping_time = round(time.time() - start_time, 2)
            print(f"✅ Scraped {len(review_texts)} reviews in {scraping_time} seconds")
            
            # Store reviews in database if available
            if use_database and self.database and product_id and scraping_result.get('reviews'):
                try:
                    stored_count = self.database.store_reviews(
                        product_id, 
                        scraping_result['reviews'], 
                        platform
                    )
                    print(f"💾 Stored {stored_count} reviews in database")
                    
                    # Store scraping session metadata
                    self.database.store_scraping_session(
                        product_id,
                        platform,
                        len(review_texts),
                        scraping_time,
                        "success"
                    )
                except Exception as e:
                    print(f"⚠️ Failed to store reviews in database: {e}")
            
            # Analyze the scraped reviews
            print("🧠 Starting analysis of scraped reviews...")
            analysis_result = self.analyze_customer_feedback(
                review_texts,
                include_topics=include_topics,
                include_summary=include_summary
            )
            
            # Combine results
            combined_result = {
                'product_name': product_name,
                'product_id': product_id,
                'platform': platform,
                'scraping_time': scraping_time,
                'analysis_time': analysis_result.get('processing_time', 0),
                'total_time': scraping_time + analysis_result.get('processing_time', 0),
                'reviews_found': len(review_texts),
                'database_stored': use_database and self.database and product_id is not None,
                'scraping_details': {
                    'sources_attempted': scraping_result.get('sources_attempted', [platform]),
                    'sources_successful': scraping_result.get('sources_successful', [platform] if scraping_result.get('success') else []),
                    'total_reviews': scraping_result.get('total', 0)
                },
                'analysis': analysis_result,
                'sample_reviews': review_texts[:3],  # Include first 3 reviews as samples
                'success': True
            }
            
            # Store analysis results in database if available
            if use_database and self.database and product_id:
                try:
                    self.database.store_analysis_results(product_id, analysis_result)
                    print("💾 Stored analysis results in database")
                except Exception as e:
                    print(f"⚠️ Failed to store analysis in database: {e}")
            
            print(f"🎯 Complete analysis finished in {combined_result['total_time']} seconds")
            return combined_result
            
        except Exception as e:
            return {
                'error': f'Error during scraping and analysis: {str(e)}',
                'product_name': product_name,
                'platform': platform,
                'success': False,
                'scraping_time': round(time.time() - start_time, 2)
            }
    
    def get_stored_product_reviews(self, product_name: str) -> Dict:
        """
        Get previously stored reviews for a product from database
        
        Args:
            product_name: Name of the product to search for
            
        Returns:
            Dictionary containing stored reviews and analysis
        """
        if not self.database:
            return {
                'error': 'Database not available',
                'success': False
            }
        
        try:
            # Search for products
            products = self.database.search_products(product_name)
            
            if not products:
                return {
                    'error': f'No stored data found for "{product_name}"',
                    'success': False
                }
            
            # Get the most recent product
            product = products[0]
            product_id = product['id']
            
            # Get reviews
            reviews = self.database.get_product_reviews(product_id)
            
            # Get analytics
            analytics = self.database.get_product_analytics(product_id)
            
            return {
                'product': product,
                'reviews': reviews,
                'analytics': analytics,
                'total_reviews': len(reviews),
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Error retrieving stored data: {str(e)}',
                'success': False
            }
    
    def analyze_stored_reviews(self, product_name: str) -> Dict:
        """
        Analyze previously stored reviews for a product
        
        Args:
            product_name: Name of the product
            
        Returns:
            Analysis results for stored reviews
        """
        stored_data = self.get_stored_product_reviews(product_name)
        
        if not stored_data.get('success'):
            return stored_data
        
        # Extract review texts
        review_texts = [review['review_text'] for review in stored_data['reviews'] if review.get('review_text')]
        
        if not review_texts:
            return {
                'error': 'No review texts found in stored data',
                'success': False
            }
        
        # Analyze the stored reviews
        analysis_result = self.analyze_customer_feedback(review_texts)
        
        return {
            'product_name': product_name,
            'source': 'database',
            'reviews_analyzed': len(review_texts),
            'analysis': analysis_result,
            'stored_data': stored_data,
            'success': True
        }
    
    def list_stored_products(self) -> List[Dict]:
        """
        List all products stored in database
        
        Returns:
            List of stored products
        """
        if not self.database:
            return []
        
        try:
            # This would need to be implemented in the database class
            # For now, return empty list
            return []
        except Exception as e:
            print(f"Error listing products: {e}")
            return []
    
    def get_system_status(self) -> Dict:
        """Get status of all system components"""
        return {
            'sentiment_analyzer': self.sentiment_analyzer is not None,
            'topic_extractor': self.topic_extractor is not None,
            'text_summarizer': self.text_summarizer is not None,
            'review_scraper': self.review_scraper is not None,
            'database': self.database is not None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

def demo_analysis():
    """Demonstration of the complete system"""
    print("🎯 Starting Customer Sentiment Summarizer Demo\n")
    
    # Sample customer reviews
    sample_reviews = [
        "I absolutely love this product! The battery life is amazing and it works perfectly. Highly recommended!",
        "Terrible experience. The product broke after just one week and customer service was unhelpful.",
        "Decent product for the price. Battery life could be better but overall satisfied with the purchase.",
        "Fast delivery and the product quality exceeded my expectations. Great camera and display!",
        "Not worth the money. Poor build quality and the battery drains too quickly."
    ]
    
    try:
        # Initialize the system
        analyzer = CustomerSentimentSummarizer()
        
        # Show system status
        status = analyzer.get_system_status()
        print("🔧 System Status:")
        for component, available in status.items():
            if component != 'timestamp':
                print(f"  {component}: {'✅' if available else '❌'}")
        print()
        
        # Analyze single review
        print("🔍 Single Review Analysis:")
        single_result = analyzer.analyze_customer_feedback(sample_reviews[0])
        print(f"Sentiment: {single_result['sentiment']['overall_sentiment']}")
        print(f"Summary: {single_result['summary']['text'][:100]}...")
        print()
        
        # Analyze multiple reviews
        print("🔍 Multiple Reviews Analysis:")
        multi_result = analyzer.analyze_customer_feedback(sample_reviews)
        print(f"Overall Sentiment: {multi_result['sentiment']['overall_sentiment']}")
        print(f"Processing time: {multi_result['processing_time']} seconds")
        print(f"Components used: {', '.join(multi_result['components_used'])}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    demo_analysis()

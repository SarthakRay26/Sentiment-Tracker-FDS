"""
Main Backend Integration Module
Integrates sentiment analysis, topic extraction, and summarization
"""

import os
import sys
from typing import Dict, List, Union, Optional
import json
import time

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Sentiment analyzer import error: {e}")
    SENTIMENT_AVAILABLE = False

try:
    from topics import TopicExtractor
    TOPICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Topic extractor import error: {e}")
    print("Continuing without topic extraction...")
    TOPICS_AVAILABLE = False

try:
    from summarizer import TextSummarizer
    SUMMARIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Summarizer import error: {e}")
    print("Continuing without summarization...")
    SUMMARIZER_AVAILABLE = False

class CustomerSentimentSummarizer:
    def __init__(self, 
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 summarization_model: str = "facebook/bart-large-cnn",
                 use_bertopic: bool = True,
                 num_topics: int = 5):
        """
        Initialize the complete sentiment summarizer system
        
        Args:
            sentiment_model: HuggingFace model for sentiment analysis
            summarization_model: HuggingFace model for text summarization
            use_bertopic: Whether to use BERTopic for topic modeling
            num_topics: Number of topics to extract
        """
        self.sentiment_analyzer = None
        self.topic_extractor = None
        self.text_summarizer = None
        
        print("🚀 Initializing Customer Sentiment Summarizer...")
        
        # Initialize components
        self._initialize_components(sentiment_model, summarization_model, use_bertopic, num_topics)
        
        print("✅ Customer Sentiment Summarizer ready!")
    
    def _initialize_components(self, sentiment_model, summarization_model, use_bertopic, num_topics):
        """Initialize all AI components"""
        # Initialize components with availability checks
        try:
            if SENTIMENT_AVAILABLE:
                print("📊 Loading sentiment analysis model...")
                self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
            else:
                print("⚠️ Sentiment analysis not available")
                
            if TOPICS_AVAILABLE:
                print("🔍 Loading topic extraction model...")
                self.topic_extractor = TopicExtractor(use_bertopic=use_bertopic, num_topics=num_topics)
            else:
                print("⚠️ Topic extraction not available")
                
            if SUMMARIZER_AVAILABLE:
                print("📝 Loading text summarization model...")
                self.text_summarizer = TextSummarizer(summarization_model)
            else:
                print("⚠️ Text summarization not available")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            # Don't raise, continue with available components
    
    def analyze_customer_feedback(self, 
                                text_input: Union[str, List[str]], 
                                include_topics: bool = True,
                                include_summary: bool = True,
                                max_summary_length: int = 150) -> Dict:
        """
        Complete analysis of customer feedback including sentiment, topics, and summary
        
        Args:
            text_input: Single review string or list of review strings
            include_topics: Whether to extract topics
            include_summary: Whether to generate summary
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
            'processing_time': 0
        }
        
        try:
            # 1. Sentiment Analysis
            if self.sentiment_analyzer:
                print("📊 Analyzing sentiment...")
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(texts)
                results['sentiment'] = self._format_sentiment_results(sentiment_result, single_input)
            else:
                results['sentiment'] = {'error': 'Sentiment analyzer not available'}
            
            # 2. Topic Extraction
            if include_topics and self.topic_extractor:
                print("🔍 Extracting topics...")
                topic_result = self.topic_extractor.extract_topics(texts)
                results['topics'] = self._format_topic_results(topic_result)
            elif include_topics:
                results['topics'] = {'error': 'Topic extractor not available', 'topics': []}
            
            # 3. Text Summarization
            if include_summary and self.text_summarizer:
                print("📝 Generating summary...")
                if single_input:
                    summary_result = self.text_summarizer.summarize_text(texts[0], max_length=max_summary_length)
                else:
                    summary_result = self.text_summarizer.summarize_multiple_reviews(texts, max_length=max_summary_length)
                results['summary'] = self._format_summary_results(summary_result)
            elif include_summary:
                results['summary'] = {'error': 'Text summarizer not available', 'text': 'Summary not available', 'success': False}
            
            # Calculate processing time
            results['processing_time'] = round(time.time() - start_time, 2)
            
            # Add insights
            results['insights'] = self._generate_insights(results)
            
            print(f"✅ Analysis complete in {results['processing_time']} seconds")
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
            compression = results['summary']['compression_ratio']
            if compression < 0.3:
                insights['summary_quality'] = 'Highly compressed'
            elif compression < 0.6:
                insights['summary_quality'] = 'Well compressed'
            else:
                insights['summary_quality'] = 'Lightly compressed'
        
        return insights
    
    def batch_analyze(self, text_list: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Analyze multiple texts in batches
        
        Args:
            text_list: List of texts to analyze
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            print(f"📊 Processing batch {i//batch_size + 1}/{(len(text_list) + batch_size - 1)//batch_size}")
            
            batch_result = self.analyze_customer_feedback(batch)
            results.append(batch_result)
        
        return results
    
    def save_results(self, results: Dict, filepath: str):
        """Save analysis results to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

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
        
        # Analyze single review
        print("\n🔍 Single Review Analysis:")
        single_result = analyzer.analyze_customer_feedback(sample_reviews[0])
        print(f"Sentiment: {single_result['sentiment']['overall_sentiment']}")
        print(f"Summary: {single_result['summary']['text']}")
        
        # Analyze multiple reviews
        print("\n🔍 Multiple Reviews Analysis:")
        multi_result = analyzer.analyze_customer_feedback(sample_reviews)
        print(f"Overall Sentiment: {multi_result['sentiment']['overall_sentiment']}")
        print(f"Topics Found: {[topic['label'] for topic in multi_result['topics']['topics']]}")
        print(f"Summary: {multi_result['summary']['text']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    demo_analysis()

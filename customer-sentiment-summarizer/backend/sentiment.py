"""
Sentiment Analysis Module
Uses DistilBERT for sentiment classification
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from typing import Dict, List, Union

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer with DistilBERT model
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            print(f"✅ Sentiment model '{self.model_name}' loaded successfully")
        except Exception as e:
            print(f"❌ Error loading sentiment model: {e}")
            raise
    
    def analyze_sentiment(self, text: Union[str, List[str]]) -> Dict:
        """
        Analyze sentiment of given text(s)
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            results = []
            for single_text in text:
                # Get predictions
                predictions = self.pipeline(single_text)
                
                # Process results
                sentiment_scores = {}
                for pred in predictions[0]:
                    label = pred['label'].lower()
                    score = pred['score']
                    sentiment_scores[label] = score
                
                # Determine primary sentiment
                primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
                
                # Calculate confidence and create readable output
                confidence = primary_sentiment[1]
                sentiment_label = primary_sentiment[0]
                
                # Map to more readable labels
                readable_sentiment = self._map_sentiment_label(sentiment_label, confidence)
                
                result = {
                    'text': single_text,
                    'sentiment': readable_sentiment,
                    'confidence': confidence,
                    'scores': sentiment_scores,
                    'raw_prediction': predictions[0]
                }
                results.append(result)
            
            # If single text, return single result, otherwise return list
            if len(results) == 1:
                return results[0]
            
            return {
                'individual_results': results,
                'overall_sentiment': self._calculate_overall_sentiment(results)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
            return {
                'error': str(e),
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {}
            }
    
    def _map_sentiment_label(self, label: str, confidence: float) -> str:
        """
        Map model labels to readable sentiment labels
        
        Args:
            label: Original model label
            confidence: Confidence score
            
        Returns:
            Readable sentiment label
        """
        # Mapping for DistilBERT sentiment model
        label_mapping = {
            'positive': 'Positive',
            'negative': 'Negative',
            'neutral': 'Neutral'
        }
        
        # If confidence is low, classify as neutral
        if confidence < 0.6:
            return 'Neutral'
        
        return label_mapping.get(label, 'Neutral')
    
    def _calculate_overall_sentiment(self, results: List[Dict]) -> Dict:
        """
        Calculate overall sentiment from multiple text results
        
        Args:
            results: List of individual sentiment results
            
        Returns:
            Overall sentiment summary
        """
        if not results:
            return {'sentiment': 'Neutral', 'confidence': 0.0}
        
        # Aggregate scores
        total_positive = sum(r.get('scores', {}).get('positive', 0) for r in results)
        total_negative = sum(r.get('scores', {}).get('negative', 0) for r in results)
        
        avg_positive = total_positive / len(results)
        avg_negative = total_negative / len(results)
        
        # Determine overall sentiment
        if abs(avg_positive - avg_negative) < 0.1:
            overall_sentiment = 'Mixed'
            confidence = 0.5
        elif avg_positive > avg_negative:
            overall_sentiment = 'Positive'
            confidence = avg_positive
        else:
            overall_sentiment = 'Negative'
            confidence = avg_negative
        
        return {
            'sentiment': overall_sentiment,
            'confidence': confidence,
            'positive_score': avg_positive,
            'negative_score': avg_negative,
            'distribution': {
                'positive': avg_positive,
                'negative': avg_negative,
                'neutral': 1 - (avg_positive + avg_negative)
            }
        }

def test_sentiment_analyzer():
    """Test function for the sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "The battery life is excellent but the screen quality is poor."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print(f"Scores: {result['scores']}")
        print("-" * 50)

if __name__ == "__main__":
    test_sentiment_analyzer()

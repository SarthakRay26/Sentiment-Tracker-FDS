"""
Text Summarization Module
Uses BART-large-CNN for summarization with fallback to T5-small
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Dict, Union
import re

class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", fallback_model: str = "t5-small"):
        """
        Initialize the text summarizer
        
        Args:
            model_name: Primary model for summarization (BART-large-CNN)
            fallback_model: Fallback model if primary fails (T5-small)
        """
        self.primary_model = model_name
        self.fallback_model = fallback_model
        self.current_model = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model with fallback logic"""
        try:
            # Try to load primary model (BART)
            self._load_specific_model(self.primary_model)
            self.current_model = self.primary_model
            print(f"✅ Primary summarization model '{self.primary_model}' loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load primary model '{self.primary_model}': {e}")
            try:
                # Fallback to T5-small
                self._load_specific_model(self.fallback_model)
                self.current_model = self.fallback_model
                print(f"✅ Fallback summarization model '{self.fallback_model}' loaded successfully")
            except Exception as e2:
                print(f"❌ Failed to load fallback model '{self.fallback_model}': {e2}")
                raise RuntimeError("Could not load any summarization model")
    
    def _load_specific_model(self, model_name: str):
        """Load a specific model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline based on model type
        if "bart" in model_name.lower():
            self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        elif "t5" in model_name.lower():
            self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        else:
            self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
    
    def summarize_text(self, text: Union[str, List[str]], max_length: int = 150, min_length: int = 30) -> Dict:
        """
        Summarize given text(s)
        
        Args:
            text: Single text string or list of texts to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary containing summarization results
        """
        if isinstance(text, list):
            # Join multiple texts
            combined_text = " ".join(text)
        else:
            combined_text = text
        
        # Preprocess text
        processed_text = self._preprocess_text(combined_text)
        
        if not processed_text or len(processed_text.split()) < 10:
            return {
                'summary': "Text too short to summarize effectively.",
                'original_length': len(combined_text.split()),
                'summary_length': 0,
                'compression_ratio': 0,
                'model_used': self.current_model,
                'success': False
            }
        
        try:
            # Adjust parameters based on text length
            text_length = len(processed_text.split())
            adjusted_max_length = min(max_length, max(text_length // 3, min_length))
            adjusted_min_length = min(min_length, adjusted_max_length // 2)
            
            if "t5" in self.current_model.lower():
                # For T5, add prefix
                input_text = f"summarize: {processed_text}"
                result = self.pipeline(
                    input_text,
                    max_length=adjusted_max_length,
                    min_length=adjusted_min_length,
                    do_sample=False
                )
                summary = result[0]['generated_text']
            else:
                # For BART and other models
                result = self.pipeline(
                    processed_text,
                    max_length=adjusted_max_length,
                    min_length=adjusted_min_length,
                    do_sample=False
                )
                summary = result[0]['summary_text']
            
            # Post-process summary
            summary = self._postprocess_summary(summary)
            
            return {
                'summary': summary,
                'original_text': combined_text,
                'original_length': text_length,
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / text_length,
                'model_used': self.current_model,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            
            # Fallback to extractive summarization
            fallback_summary = self._extractive_fallback(processed_text, max_length)
            
            return {
                'summary': fallback_summary,
                'original_text': combined_text,
                'original_length': len(combined_text.split()),
                'summary_length': len(fallback_summary.split()),
                'compression_ratio': len(fallback_summary.split()) / len(combined_text.split()),
                'model_used': 'extractive_fallback',
                'error': str(e),
                'success': False
            }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for summarization
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Ensure text doesn't exceed model's max length (approximate)
        max_tokens = 1024  # Conservative estimate for most models
        words = text.split()
        if len(words) > max_tokens:
            text = ' '.join(words[:max_tokens])
        
        return text
    
    def _postprocess_summary(self, summary: str) -> str:
        """
        Post-process the generated summary
        
        Args:
            summary: Raw summary from model
            
        Returns:
            Cleaned summary
        """
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary.strip())
        
        # Ensure proper capitalization
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        return summary
    
    def _extractive_fallback(self, text: str, max_length: int) -> str:
        """
        Simple extractive summarization as fallback
        
        Args:
            text: Input text
            max_length: Maximum length of summary
            
        Returns:
            Extractive summary
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return "Unable to generate summary."
        
        # Simple scoring: prefer sentences with certain keywords
        keywords = ['good', 'bad', 'excellent', 'terrible', 'love', 'hate', 'recommend', 'quality', 'price', 'fast', 'slow']
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Build summary within length limit
        summary_parts = []
        current_length = 0
        target_length = max_length // 2  # Be conservative with extractive summary
        
        for score, sentence in scored_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= target_length:
                summary_parts.append(sentence)
                current_length += sentence_length
            if current_length >= target_length:
                break
        
        if not summary_parts:
            # If no sentences fit, just take the first sentence
            summary_parts = [sentences[0]]
        
        return '. '.join(summary_parts) + '.'
    
    def summarize_multiple_reviews(self, reviews: List[str], max_length: int = 200) -> Dict:
        """
        Summarize multiple customer reviews into a cohesive summary
        
        Args:
            reviews: List of customer review texts
            max_length: Maximum length of combined summary
            
        Returns:
            Dictionary containing summarization results
        """
        if not reviews:
            return {
                'summary': "No reviews provided.",
                'review_count': 0,
                'success': False
            }
        
        # Group reviews by sentiment if possible (basic approach)
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []
        
        for review in reviews:
            # Simple sentiment detection for grouping
            positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'perfect', 'satisfied']
            negative_words = ['bad', 'terrible', 'hate', 'awful', 'disappointed', 'poor', 'worst']
            
            review_lower = review.lower()
            pos_count = sum(1 for word in positive_words if word in review_lower)
            neg_count = sum(1 for word in negative_words if word in review_lower)
            
            if pos_count > neg_count:
                positive_reviews.append(review)
            elif neg_count > pos_count:
                negative_reviews.append(review)
            else:
                neutral_reviews.append(review)
        
        # Summarize each group
        summaries = []
        
        if positive_reviews:
            pos_summary = self.summarize_text(positive_reviews, max_length=max_length//3)
            if pos_summary['success']:
                summaries.append(f"Positive feedback: {pos_summary['summary']}")
        
        if negative_reviews:
            neg_summary = self.summarize_text(negative_reviews, max_length=max_length//3)
            if neg_summary['success']:
                summaries.append(f"Negative feedback: {neg_summary['summary']}")
        
        if neutral_reviews:
            neu_summary = self.summarize_text(neutral_reviews, max_length=max_length//3)
            if neu_summary['success']:
                summaries.append(f"Mixed feedback: {neu_summary['summary']}")
        
        # Combine summaries
        if summaries:
            combined_summary = " ".join(summaries)
        else:
            # Fallback: summarize all reviews together
            all_text = " ".join(reviews)
            result = self.summarize_text(all_text, max_length=max_length)
            combined_summary = result['summary']
        
        return {
            'summary': combined_summary,
            'review_count': len(reviews),
            'positive_count': len(positive_reviews),
            'negative_count': len(negative_reviews),
            'neutral_count': len(neutral_reviews),
            'model_used': self.current_model,
            'success': True
        }

def test_summarizer():
    """Test function for the text summarizer"""
    summarizer = TextSummarizer()
    
    test_text = """
    I recently purchased this smartphone and have been using it for about two weeks now. 
    Overall, I'm quite impressed with the performance and build quality. The battery life 
    is exceptional - it easily lasts a full day even with heavy usage including gaming, 
    video streaming, and photography. The camera quality is outstanding, especially in 
    good lighting conditions. Photos are sharp, colors are vibrant, and the night mode 
    actually works quite well. The display is bright and clear with excellent color 
    reproduction. However, I did notice that the phone tends to get a bit warm during 
    intensive tasks like gaming or video recording. The price point is reasonable 
    considering all the features you get. The build quality feels premium and the 
    design is sleek. I would definitely recommend this phone to anyone looking for 
    a reliable device with great camera capabilities and long battery life.
    """
    
    result = summarizer.summarize_text(test_text)
    
    print("📝 Original text length:", result['original_length'], "words")
    print("📋 Summary length:", result['summary_length'], "words")
    print("📊 Compression ratio:", f"{result['compression_ratio']:.2f}")
    print("🤖 Model used:", result['model_used'])
    print("✅ Success:", result['success'])
    print("\n📝 Summary:")
    print(result['summary'])

if __name__ == "__main__":
    test_summarizer()

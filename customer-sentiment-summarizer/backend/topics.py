"""
Topic Modeling Module
Uses BERTopic for topic extraction with fallback to Gensim LDA
"""

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("⚠️ BERTopic not available, will use Gensim LDA as fallback")

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union
import warnings
warnings.filterwarnings("ignore")

# Fallback imports for Gensim LDA
try:
    from gensim import corpora, models
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Gensim import error: {e}")
    GENSIM_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Gensim compatibility error: {e}")
    GENSIM_AVAILABLE = False

class TopicExtractor:
    def __init__(self, use_bertopic: bool = True, num_topics: int = 5):
        """
        Initialize the topic extractor
        
        Args:
            use_bertopic: Whether to use BERTopic (True) or Gensim LDA (False)
            num_topics: Number of topics to extract
        """
        self.use_bertopic = use_bertopic and BERTOPIC_AVAILABLE
        self.num_topics = num_topics
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize the appropriate model
        if self.use_bertopic:
            self._initialize_bertopic()
        else:
            self._initialize_lda()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _initialize_bertopic(self):
        """Initialize BERTopic model"""
        try:
            # Use a smaller sentence transformer model for efficiency
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model = BERTopic(
                embedding_model=sentence_model,
                nr_topics=self.num_topics,
                verbose=False
            )
            print("✅ BERTopic model initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing BERTopic: {e}")
            print("🔄 Falling back to Gensim LDA")
            self.use_bertopic = False
            self._initialize_lda()
    
    def _initialize_lda(self):
        """Initialize Gensim LDA model (fallback)"""
        if not GENSIM_AVAILABLE:
            print("❌ Neither BERTopic nor Gensim is available")
            return
        
        print("✅ Gensim LDA initialized (will be trained on input data)")
    
    def preprocess_text(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for topic modeling
        
        Args:
            texts: List of text documents
            
        Returns:
            List of preprocessed token lists
        """
        stop_words = set(stopwords.words('english'))
        processed_texts = []
        
        for text in texts:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short words
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in stop_words and len(token) > 2
            ]
            
            processed_texts.append(tokens)
        
        return processed_texts
    
    def extract_topics(self, texts: Union[str, List[str]]) -> Dict:
        """
        Extract topics from given texts
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Dictionary containing topic extraction results
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return {'error': 'No texts provided', 'topics': []}
        
        try:
            if self.use_bertopic:
                return self._extract_topics_bertopic(texts)
            else:
                return self._extract_topics_lda(texts)
        except Exception as e:
            print(f"❌ Error extracting topics: {e}")
            return {
                'error': str(e),
                'topics': [],
                'method': 'bertopic' if self.use_bertopic else 'lda'
            }
    
    def _extract_topics_bertopic(self, texts: List[str]) -> Dict:
        """Extract topics using BERTopic"""
        # Fit the model on the texts
        topics, probabilities = self.model.fit_transform(texts)
        
        # Get topic information
        topic_info = self.model.get_topic_info()
        
        # Extract readable topic labels
        topic_labels = []
        for topic_id in range(len(topic_info)):
            if topic_id == 0:  # Skip outlier topic (-1 mapped to 0)
                continue
            
            topic_words = self.model.get_topic(topic_id - 1)  # BERTopic uses -1 indexing
            if topic_words:
                # Create readable label from top words
                top_words = [word for word, _ in topic_words[:3]]
                label = self._create_topic_label(top_words)
                topic_labels.append({
                    'id': topic_id - 1,
                    'label': label,
                    'words': topic_words[:5],
                    'size': topic_info.iloc[topic_id]['Count'] if topic_id < len(topic_info) else 0
                })
        
        return {
            'topics': topic_labels,
            'method': 'bertopic',
            'document_topics': [{'document_id': i, 'topic_id': topic, 'probability': prob} 
                              for i, (topic, prob) in enumerate(zip(topics, probabilities))],
            'success': True
        }
    
    def _extract_topics_lda(self, texts: List[str]) -> Dict:
        """Extract topics using Gensim LDA"""
        if not GENSIM_AVAILABLE:
            return {'error': 'Gensim not available', 'topics': []}
        
        # Preprocess texts
        processed_texts = self.preprocess_text(texts)
        
        # Filter out empty documents
        processed_texts = [doc for doc in processed_texts if doc]
        
        if not processed_texts:
            return {'error': 'No valid documents after preprocessing', 'topics': []}
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(doc) for doc in processed_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=min(self.num_topics, len(processed_texts)),
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = lda_model.print_topics(num_words=5)
        topic_labels = []
        
        for topic_id, topic_words in topics:
            # Parse topic words
            words = re.findall(r'"([^"]*)"', topic_words)
            label = self._create_topic_label(words[:3])
            
            topic_labels.append({
                'id': topic_id,
                'label': label,
                'words': words[:5],
                'raw': topic_words
            })
        
        return {
            'topics': topic_labels,
            'method': 'lda',
            'success': True
        }
    
    def _create_topic_label(self, top_words: List[str]) -> str:
        """
        Create a readable topic label from top words
        
        Args:
            top_words: List of top words for the topic
            
        Returns:
            Readable topic label
        """
        if not top_words:
            return "Unknown Topic"
        
        # Common patterns for creating meaningful labels
        word_mappings = {
            'battery': 'Battery Performance',
            'screen': 'Display Quality',
            'camera': 'Camera Quality',
            'price': 'Pricing',
            'delivery': 'Shipping & Delivery',
            'service': 'Customer Service',
            'quality': 'Product Quality',
            'design': 'Design & Appearance',
            'performance': 'Performance',
            'sound': 'Audio Quality',
            'fast': 'Speed & Performance',
            'slow': 'Performance Issues',
            'expensive': 'Pricing Concerns',
            'cheap': 'Value for Money'
        }
        
        # Check if any top word matches our mappings
        for word in top_words:
            if word.lower() in word_mappings:
                return word_mappings[word.lower()]
        
        # If no mapping found, create label from top words
        if len(top_words) >= 2:
            return f"{top_words[0].title()} & {top_words[1].title()}"
        else:
            return f"{top_words[0].title()} Related"

def test_topic_extractor():
    """Test function for the topic extractor"""
    extractor = TopicExtractor(num_topics=3)
    
    test_texts = [
        "The battery life is excellent and lasts all day long.",
        "The camera quality is amazing with great photo clarity.",
        "Fast delivery and excellent customer service experience.",
        "The screen is bright and colors are vivid.",
        "Battery drains too quickly, very disappointed.",
        "Poor camera performance in low light conditions."
    ]
    
    result = extractor.extract_topics(test_texts)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Topic extraction successful using {result['method']}")
        print(f"Found {len(result['topics'])} topics:")
        
        for topic in result['topics']:
            print(f"  - {topic['label']} (ID: {topic['id']})")
            if 'words' in topic:
                words = [word[0] if isinstance(word, tuple) else word for word in topic['words']]
                print(f"    Words: {', '.join(words[:3])}")

if __name__ == "__main__":
    test_topic_extractor()

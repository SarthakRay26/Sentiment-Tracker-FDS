# Customer Sentiment Summarizer Chatbot 🎯

A comprehensive AI-powered chatbot that analyzes customer reviews to provide sentiment analysis, topic extraction, and intelligent summarization. Built with state-of-the-art NLP models and a user-friendly Gradio interface.

## 🌟 Features

- **Sentiment Analysis**: Uses DistilBERT to classify sentiment as positive, negative, neutral, or mixed with confidence scores
- **Topic Modeling**: Employs BERTopic (with Gensim LDA fallback) to extract key discussion topics from reviews
- **Text Summarization**: Leverages BART-large-CNN (with T5-small fallback) to generate concise, readable summaries
- **Interactive Web Interface**: Gradio-powered chatbot interface with visualizations
- **Batch Processing**: Analyze single reviews or multiple reviews simultaneously
- **Visual Analytics**: Charts and graphs showing sentiment distribution and scores
- **Analysis History**: Track and review past analyses

## 🏗️ Project Structure

```
customer-sentiment-summarizer/
├── backend/
│   ├── sentiment.py       # Sentiment analysis using DistilBERT
│   ├── topics.py          # Topic modeling with BERTopic/Gensim LDA
│   ├── summarizer.py      # Text summarization with BART/T5
│   ├── main.py            # Main integration module
│   └── __init__.py
├── frontend/
│   ├── app.py             # Gradio web interface
│   └── __init__.py
├── data/
│   └── sample_reviews.txt # Sample dataset for testing
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd customer-sentiment-summarizer

# Install dependencies
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Run the Application

```bash
# Start the Gradio interface
python frontend/app.py
```

The application will be available at `http://localhost:7860`

## 💻 Usage

### Single Review Analysis

1. Navigate to the "Single Review Analysis" tab
2. Paste a customer review in the text box
3. Click "Analyze Review"
4. View sentiment, topics, and summary results

### Multiple Reviews Analysis

1. Go to the "Multiple Reviews Analysis" tab
2. Enter multiple reviews (one per line)
3. Click "Analyze Reviews"
4. Review aggregated sentiment distribution and insights

### Example Review

```
I absolutely love this smartphone! The battery life is incredible and easily 
lasts all day. The camera quality is outstanding, especially the night mode. 
Fast delivery and excellent packaging. Highly recommend!
```

**Expected Output:**
- **Sentiment**: Positive (95% confidence)
- **Topics**: Battery Performance, Camera Quality, Delivery & Service
- **Summary**: Customer highly satisfied with smartphone's battery life, camera quality, and delivery experience.

## 🧠 AI Models Used

### Sentiment Analysis
- **Primary**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Features**: Binary sentiment classification with confidence scores
- **Output**: Positive/Negative/Neutral with probability distributions

### Topic Modeling
- **Primary**: BERTopic with `all-MiniLM-L6-v2` embeddings
- **Fallback**: Gensim LDA
- **Features**: Automatic topic discovery with readable labels

### Text Summarization
- **Primary**: `facebook/bart-large-cnn`
- **Fallback**: `t5-small`
- **Features**: Abstractive summarization with compression ratio tracking

## 📊 API Reference

### Backend Classes

#### `SentimentAnalyzer`
```python
from backend.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment("Great product!")
```

#### `TopicExtractor`
```python
from backend.topics import TopicExtractor

extractor = TopicExtractor(use_bertopic=True, num_topics=5)
topics = extractor.extract_topics(["Review 1", "Review 2"])
```

#### `TextSummarizer`
```python
from backend.summarizer import TextSummarizer

summarizer = TextSummarizer()
summary = summarizer.summarize_text("Long review text...")
```

#### `CustomerSentimentSummarizer` (Main Class)
```python
from backend.main import CustomerSentimentSummarizer

analyzer = CustomerSentimentSummarizer()
result = analyzer.analyze_customer_feedback("Customer review text")
```

### Result Format

```python
{
    'sentiment': {
        'overall_sentiment': 'Positive',
        'confidence': 0.95,
        'scores': {'positive': 0.95, 'negative': 0.05}
    },
    'topics': {
        'topics': [
            {'label': 'Battery Performance', 'keywords': ['battery', 'life', 'power']},
            {'label': 'Camera Quality', 'keywords': ['camera', 'photo', 'quality']}
        ]
    },
    'summary': {
        'text': 'Customer highly satisfied with product performance...',
        'compression_ratio': 0.25
    },
    'processing_time': 2.3
}
```

## 🛠️ Configuration

### Model Configuration

You can customize the models used by modifying the initialization parameters:

```python
analyzer = CustomerSentimentSummarizer(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
    summarization_model="facebook/bart-large-cnn",
    use_bertopic=True,
    num_topics=5
)
```

### Environment Variables

Create a `.env` file for configuration:

```env
HUGGINGFACE_HUB_CACHE=./models
TRANSFORMERS_CACHE=./models
TOKENIZERS_PARALLELISM=false
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- 2GB+ disk space for models

### Python Dependencies
See `requirements.txt` for complete list. Key dependencies:
- `transformers>=4.20.0`
- `gradio>=3.40.0`
- `torch>=1.9.0`
- `bertopic>=0.15.0`
- `nltk>=3.8`

## 🚀 Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload your code
3. Set the SDK to "Gradio"
4. The app will automatically deploy

### Local Deployment

```bash
# Using gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker frontend.app:main

# Using Docker (create Dockerfile)
docker build -t sentiment-analyzer .
docker run -p 7860:7860 sentiment-analyzer
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure stable internet connection for first-time model downloads
   - Check available disk space (models require ~2GB)

2. **Memory Issues**
   - Reduce batch size for large datasets
   - Use smaller models (T5-small instead of BART-large)

3. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

4. **Gradio Interface Not Loading**
   - Check if port 7860 is available
   - Try different port: `interface.launch(server_port=8080)`

### Performance Optimization

- **CPU Only**: Models will run on CPU by default
- **GPU Acceleration**: Install CUDA-compatible PyTorch
- **Model Caching**: Models are cached after first download
- **Batch Processing**: Use batch analysis for multiple reviews

## 🧪 Testing

Run the test functions:

```bash
# Test individual components
python backend/sentiment.py
python backend/topics.py
python backend/summarizer.py

# Test complete system
python backend/main.py
```

## 📈 Performance Metrics

### Typical Processing Times
- Single review analysis: 2-5 seconds
- Multiple reviews (5-10): 5-15 seconds
- Topic extraction: 1-3 seconds
- Summarization: 2-8 seconds

### Model Accuracy
- Sentiment Analysis: ~90% accuracy on standard datasets
- Topic Modeling: Coherence varies by domain
- Summarization: ROUGE scores comparable to state-of-the-art

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing pre-trained models and transformers library
- **Gradio** for the intuitive web interface framework
- **BERTopic** for advanced topic modeling capabilities
- **NLTK** and **Gensim** for text processing utilities

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include error logs and system information

---

**Built with ❤️ using state-of-the-art AI models and modern web technologies.**

# 🌊 Sentiment Tracker - AI-Powered Customer Review Analysis Platform

> Transform customer feedback into actionable insights with cutting-edge AI technology

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

## ✨ Features

### 🎯 Core Functionality
- **🔍 Product Review Analysis**: Analyze thousands of product reviews instantly using Kaggle datasets
- **📝 Single Review Deep Dive**: Detailed sentiment analysis of individual reviews
- **📊 Bulk Review Processing**: Upload and analyze multiple reviews simultaneously
- **🤖 AI-Powered Insights**: Intelligent summaries and trend identification

### 🧠 AI Models
- **DistilBERT**: State-of-the-art sentiment classification with confidence scores
- **BART-large-CNN**: Advanced text summarization for actionable insights
- **Apple Silicon Optimized**: MPS acceleration for M1/M2 Macs

### 🎨 Modern UI
- **Ocean Blue & Coral Theme**: Beautiful, professional interface
- **Responsive Design**: Works seamlessly across devices
- **Interactive Charts**: Visual sentiment distribution and trends
- **User-Friendly**: No technical knowledge required

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Kaggle account (optional, for dataset access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SarthakRay26/Sentiment-Tracker-FDS.git
cd Sentiment-Tracker-FDS
```

2. **Set up virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
cd customer-sentiment-summarizer
pip install -r requirements.txt
```

4. **Run the application**
```bash
python frontend/app_product_search.py
```

5. **Open in browser**
Navigate to `http://localhost:7860`

## 📊 How It Works

### 🔄 Analysis Pipeline

1. **Data Input** → User enters product name or pastes reviews
2. **Smart Dataset Selection** → AI automatically finds relevant review datasets
3. **Text Preprocessing** → Cleans and normalizes review text
4. **Sentiment Analysis** → DistilBERT classifies emotions with confidence scores
5. **Intelligent Summarization** → BART generates actionable insights
6. **Visual Results** → Beautiful charts and comprehensive reports

### 🎯 Product Categories Supported
- 📱 **Electronics**: iPhone, Samsung, MacBook, laptops, etc.
- 🍕 **Food & Beverages**: Coffee, snacks, organic products
- 🏠 **Home & Garden**: Furniture, appliances, tools
- 🎮 **Entertainment**: Games, books, toys
- 🚗 **Automotive**: Car accessories, parts
- 👔 **Fashion**: Clothing, accessories, shoes

## 🛠️ Technical Architecture

### Backend Components
```
backend/
├── main_simple.py          # Core integration module
├── sentiment.py            # DistilBERT sentiment analysis
├── summarizer.py           # BART text summarization
├── kaggle_importer.py      # Dataset integration
├── database.py             # SQLite data management
└── scraper.py              # Web scraping utilities
```

### Frontend Interface
```
frontend/
├── app_product_search.py   # Main Gradio interface
├── app_simple.py          # Alternative simple UI
└── app.py                 # Original interface
```

### Key Features
- **Multi-encoding Support**: Handles international characters (UTF-8, ISO-8859-1, etc.)
- **Smart Filtering**: Case-insensitive product matching
- **Batch Processing**: Efficient handling of large review datasets
- **Error Resilience**: Robust fallback mechanisms
- **Real-time Analysis**: Fast processing with caching

## 📈 Performance & Optimization

- **🚀 Apple Silicon**: Optimized for M1/M2 Macs with MPS acceleration
- **⚡ Fast Processing**: Most analyses complete in under 30 seconds
- **📊 Scalable**: Handles datasets with 28K+ reviews
- **🎯 Accurate**: High-confidence sentiment predictions
- **💾 Efficient**: Smart caching and memory management

## 🎨 UI Highlights

### Ocean Blue & Coral Theme
- **Professional Design**: Modern gradients and card layouts
- **Excellent Readability**: Dark text on colorful backgrounds
- **Interactive Elements**: Hover effects and smooth transitions
- **Responsive Layout**: Perfect on desktop and mobile

### User Experience
- **Intuitive Navigation**: Clear tab structure with descriptive icons
- **Helpful Guidance**: Pro tips and usage instructions
- **Real-time Feedback**: Progress indicators and status updates
- **Error Handling**: Friendly error messages and suggestions

## 📚 Usage Examples

### Analyze iPhone Reviews
```python
# Search for "iPhone" with 100 sample reviews
# Results: 77% positive, 15% negative, 8% neutral
# Insights: "Customers love camera quality but battery concerns exist"
```

### Bulk Review Analysis
```python
# Paste multiple reviews (one per line)
# Get: Sentiment distribution, individual scores, AI summary
```

### Single Review Deep Dive
```python
# Analyze: "This product exceeded my expectations!"
# Result: 94% positive confidence, detailed breakdown
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/SarthakRay26/Sentiment-Tracker-FDS.git
cd Sentiment-Tracker-FDS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the incredible Transformers library
- **Gradio**: For the amazing web interface framework
- **Kaggle**: For providing access to comprehensive datasets
- **PyTorch**: For the powerful deep learning framework

## 📞 Contact

- **Author**: Sarthak Ray
- **GitHub**: [@SarthakRay26](https://github.com/SarthakRay26)
- **Project**: [Sentiment-Tracker-FDS](https://github.com/SarthakRay26/Sentiment-Tracker-FDS)

---

<p align="center">
  <strong>Transform customer feedback into success! 🚀</strong>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/SarthakRay26">Sarthak Ray</a>
</p>

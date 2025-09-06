# 🗄️ Data Sources & Database Architecture

## **Where Reviews Are Scraped From:**

### **1. 🛒 E-commerce Platforms**
- **Amazon** (`amazon.com`)
  - Product search via search API
  - Individual product pages by ASIN
  - Customer reviews with ratings, titles, dates
  - Verified purchase indicators

- **Flipkart** (`flipkart.com`)
  - Product search and navigation
  - Review sections on product pages
  - Customer ratings and review text

- **🌐 Multiple Sources** (Combination approach)
  - Scrapes from both Amazon and Flipkart
  - Aggregates reviews from multiple platforms
  - Provides broader sentiment coverage

### **2. 🕷️ Web Scraping Technology**
- **BeautifulSoup**: HTML parsing and data extraction
- **Selenium**: Dynamic content and JavaScript-heavy sites
- **Requests**: HTTP requests with session management
- **Anti-bot measures**: User-agent rotation, rate limiting, stealth mode

---

## **🗄️ Database Integration Options**

### **Current Setup: Multi-Database Support**

#### **Option 1: SQLite (Default - No Setup Required)**
```python
database = ReviewDatabase("sqlite", "data/reviews.db")
```
- ✅ **No additional setup required**
- ✅ **File-based, portable**
- ✅ **Perfect for development and small-scale usage**
- ❌ **Single-user access**

#### **Option 2: PostgreSQL (Production Ready)**
```python
database = ReviewDatabase("postgresql", "postgresql://user:pass@localhost:5432/reviews_db")
```
- ✅ **Multi-user support**
- ✅ **High performance**
- ✅ **ACID compliance**
- ✅ **Scalable for large datasets**
- ❌ **Requires PostgreSQL installation**

#### **Option 3: MongoDB (NoSQL Option)**
```python
database = ReviewDatabase("mongodb", "mongodb://localhost:27017/")
```
- ✅ **Flexible schema**
- ✅ **JSON-native storage**
- ✅ **Horizontal scaling**
- ✅ **Great for unstructured review data**
- ❌ **Requires MongoDB installation**

---

## **📊 Database Schema**

### **Tables/Collections:**

#### **1. Products Table**
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    search_term TEXT,
    category TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **2. Reviews Table**
```sql
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    review_text TEXT NOT NULL,
    rating TEXT,
    title TEXT,
    review_date TEXT,
    verified_purchase BOOLEAN,
    source TEXT NOT NULL,  -- 'amazon', 'flipkart', etc.
    scraped_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    review_hash TEXT UNIQUE  -- Prevents duplicates
);
```

#### **3. Scraping Sessions Table**
```sql
CREATE TABLE scraping_sessions (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    source TEXT,
    reviews_found INTEGER,
    scraping_time REAL,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT,
    error_message TEXT
);
```

#### **4. Analysis Results Table**
```sql
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    overall_sentiment TEXT,
    sentiment_confidence REAL,
    sentiment_distribution TEXT,  -- JSON format
    summary_text TEXT,
    processing_time REAL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## **🔄 Data Flow Architecture**

### **Complete Workflow:**

```
1. 🔍 User Input: Product name → "iPhone 15 Pro"

2. 🗄️ Database Check: 
   └── Search for existing reviews
   └── If found → Use cached data
   └── If not found → Proceed to scraping

3. 🕷️ Web Scraping:
   ├── Search Amazon for "iPhone 15 Pro"
   ├── Find product page (ASIN: B0XXXXXXX)
   ├── Navigate to reviews section
   ├── Extract review data:
   │   ├── Review text
   │   ├── Rating (1-5 stars)
   │   ├── Review title
   │   ├── Date posted
   │   ├── Verified purchase status
   │   └── Reviewer information
   └── Repeat for other platforms

4. 💾 Database Storage:
   ├── Store product information
   ├── Store individual reviews (deduplicated)
   ├── Log scraping session metadata
   └── Track success/failure rates

5. 🧠 AI Analysis:
   ├── Sentiment Analysis (DistilBERT)
   ├── Topic Extraction (BERTopic/LDA)
   ├── Text Summarization (BART/T5)
   └── Generate insights

6. 💾 Store Analysis Results:
   ├── Overall sentiment scores
   ├── Topic distributions
   ├── Generated summaries
   └── Processing metadata

7. 📊 Present Results:
   ├── Sentiment charts
   ├── Topic visualizations
   ├── Review summaries
   └── Historical comparisons
```

---

## **💾 Data Storage Features**

### **✅ What Gets Stored:**

1. **Raw Review Data**
   - Complete review text
   - Ratings and metadata
   - Source platform information
   - Scraping timestamps

2. **Analysis Results**
   - Sentiment scores and confidence
   - Detected topics and keywords
   - Generated summaries
   - Processing performance metrics

3. **Session Metadata**
   - Scraping success/failure rates
   - Performance timing data
   - Error logs and debugging info

### **🔍 Query Capabilities:**

```python
# Get all reviews for a product
reviews = database.get_product_reviews(product_id)

# Get analytics dashboard data
analytics = database.get_product_analytics(product_id)

# Search for products
products = database.search_products("iPhone")

# Get historical sentiment trends
trends = database.get_sentiment_trends(product_id, days=30)
```

---

## **🚀 Setup Instructions**

### **Option 1: SQLite (Recommended for Testing)**
```bash
# No additional setup required!
# Database file is automatically created at: data/reviews.db
```

### **Option 2: PostgreSQL Setup**
```bash
# 1. Install PostgreSQL
brew install postgresql  # macOS
sudo apt install postgresql postgresql-contrib  # Ubuntu

# 2. Create database
createdb reviews_db

# 3. Install Python driver
pip install psycopg2-binary

# 4. Update connection string in code
DATABASE_URL = "postgresql://username:password@localhost:5432/reviews_db"
```

### **Option 3: MongoDB Setup**
```bash
# 1. Install MongoDB
brew install mongodb/brew/mongodb-community  # macOS
sudo apt install mongodb  # Ubuntu

# 2. Install Python driver
pip install pymongo

# 3. Start MongoDB service
brew services start mongodb/brew/mongodb-community  # macOS
sudo systemctl start mongodb  # Ubuntu
```

---

## **📈 Benefits of Database Integration**

### **🔄 Data Persistence**
- Reviews stored permanently
- No need to re-scrape for repeated analyses
- Historical data tracking

### **📊 Analytics & Insights**
- Trend analysis over time
- Comparative product analysis
- Performance metrics tracking

### **⚡ Performance Optimization**
- Cached results for faster responses
- Batch processing capabilities
- Reduced API calls to e-commerce sites

### **🔍 Advanced Queries**
- Filter reviews by date range
- Compare sentiment across platforms
- Track product reputation changes

---

## **🎯 Usage Examples**

### **1. First-time Product Analysis**
```python
# Scrapes fresh data and stores in database
analyzer.scrape_and_analyze_reviews("iPhone 15 Pro", platform="amazon", use_database=True)
```

### **2. Using Cached Data**
```python
# Uses stored reviews if available
stored_analysis = analyzer.analyze_stored_reviews("iPhone 15 Pro")
```

### **3. Getting Historical Data**
```python
# Retrieve all stored data for a product
product_data = analyzer.get_stored_product_reviews("iPhone 15 Pro")
```

---

## **🛡️ Data Privacy & Ethics**

### **✅ Ethical Scraping Practices:**
- Respects robots.txt files
- Implements rate limiting
- Uses public review data only
- No personal information storage
- Complies with platform terms of service

### **🔒 Data Security:**
- Local database storage
- No cloud transmission of raw data
- Encrypted connections where applicable
- Configurable data retention policies

---

**The system now provides a complete end-to-end solution from web scraping to persistent storage and analysis!** 🎉

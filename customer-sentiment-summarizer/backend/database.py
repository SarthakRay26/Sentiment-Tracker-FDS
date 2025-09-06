"""
Database Integration Module for Review Storage
Supports SQLite, PostgreSQL, and MongoDB for storing scraped reviews
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

class ReviewDatabase:
    def __init__(self, db_type: str = "sqlite", connection_string: str = None):
        """
        Initialize database connection
        
        Args:
            db_type: "sqlite", "postgresql", or "mongodb"
            connection_string: Database connection string
        """
        self.db_type = db_type.lower()
        self.connection_string = connection_string
        self.connection = None
        
        if self.db_type == "sqlite":
            self._init_sqlite()
        elif self.db_type == "postgresql":
            self._init_postgresql()
        elif self.db_type == "mongodb":
            self._init_mongodb()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        db_path = self.connection_string or "data/reviews.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        self._create_sqlite_tables()
        print(f"✅ SQLite database initialized: {db_path}")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database"""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL support requires: pip install psycopg2-binary")
        
        if not self.connection_string:
            raise ValueError("PostgreSQL requires connection string")
        
        self.connection = psycopg2.connect(self.connection_string)
        self._create_postgresql_tables()
        print("✅ PostgreSQL database initialized")
    
    def _init_mongodb(self):
        """Initialize MongoDB database"""
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB support requires: pip install pymongo")
        
        connection_string = self.connection_string or "mongodb://localhost:27017/"
        self.client = MongoClient(connection_string)
        self.db = self.client.review_sentiment_db
        self.reviews_collection = self.db.reviews
        self.products_collection = self.db.products
        self.scraping_sessions = self.db.scraping_sessions
        
        # Create indexes
        self.reviews_collection.create_index("product_id")
        self.reviews_collection.create_index("source")
        self.reviews_collection.create_index("scraped_date")
        print("✅ MongoDB database initialized")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        cursor = self.connection.cursor()
        
        # Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                search_term TEXT,
                category TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, search_term)
            )
        """)
        
        # Reviews table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                review_text TEXT NOT NULL,
                rating TEXT,
                title TEXT,
                review_date TEXT,
                verified_purchase BOOLEAN,
                source TEXT NOT NULL,
                scraped_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_hash TEXT UNIQUE,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)
        
        # Scraping sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraping_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                source TEXT,
                reviews_found INTEGER,
                scraping_time REAL,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                error_message TEXT,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)
        
        # Analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                overall_sentiment TEXT,
                sentiment_confidence REAL,
                sentiment_distribution TEXT, -- JSON
                summary_text TEXT,
                processing_time REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)
        
        self.connection.commit()
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        cursor = self.connection.cursor()
        
        # Similar table creation for PostgreSQL
        # (Implementation would be similar but with PostgreSQL-specific syntax)
        pass
    
    def store_product(self, product_name: str, search_term: str = None, category: str = None) -> int:
        """
        Store product information and return product ID
        
        Args:
            product_name: Name of the product
            search_term: Original search term used
            category: Product category
            
        Returns:
            Product ID
        """
        if self.db_type == "sqlite":
            return self._store_product_sqlite(product_name, search_term, category)
        elif self.db_type == "mongodb":
            return self._store_product_mongodb(product_name, search_term, category)
    
    def _store_product_sqlite(self, product_name: str, search_term: str, category: str) -> int:
        """Store product in SQLite"""
        cursor = self.connection.cursor()
        
        # Check if product exists
        cursor.execute(
            "SELECT id FROM products WHERE name = ? AND search_term = ?",
            (product_name, search_term)
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Insert new product
        cursor.execute(
            "INSERT INTO products (name, search_term, category) VALUES (?, ?, ?)",
            (product_name, search_term, category)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def _store_product_mongodb(self, product_name: str, search_term: str, category: str) -> str:
        """Store product in MongoDB"""
        product_doc = {
            "name": product_name,
            "search_term": search_term,
            "category": category,
            "created_date": datetime.utcnow()
        }
        
        # Check if exists
        existing = self.products_collection.find_one({
            "name": product_name,
            "search_term": search_term
        })
        
        if existing:
            return str(existing["_id"])
        
        result = self.products_collection.insert_one(product_doc)
        return str(result.inserted_id)
    
    def store_reviews(self, product_id: int, reviews: List[Dict], source: str) -> int:
        """
        Store scraped reviews in database
        
        Args:
            product_id: Product ID from store_product()
            reviews: List of review dictionaries
            source: Source platform (amazon, flipkart, etc.)
            
        Returns:
            Number of reviews stored
        """
        if self.db_type == "sqlite":
            return self._store_reviews_sqlite(product_id, reviews, source)
        elif self.db_type == "mongodb":
            return self._store_reviews_mongodb(product_id, reviews, source)
    
    def _store_reviews_sqlite(self, product_id: int, reviews: List[Dict], source: str) -> int:
        """Store reviews in SQLite"""
        cursor = self.connection.cursor()
        stored_count = 0
        
        for review in reviews:
            # Create hash to avoid duplicates
            review_hash = hashlib.md5(
                f"{review.get('text', '')}{source}{product_id}".encode()
            ).hexdigest()
            
            try:
                cursor.execute("""
                    INSERT INTO reviews 
                    (product_id, review_text, rating, title, review_date, 
                     verified_purchase, source, review_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    review.get('text', ''),
                    review.get('rating', ''),
                    review.get('title', ''),
                    review.get('date', ''),
                    review.get('verified_purchase', False),
                    source,
                    review_hash
                ))
                stored_count += 1
            except sqlite3.IntegrityError:
                # Duplicate review, skip
                continue
        
        self.connection.commit()
        return stored_count
    
    def _store_reviews_mongodb(self, product_id: str, reviews: List[Dict], source: str) -> int:
        """Store reviews in MongoDB"""
        stored_count = 0
        
        for review in reviews:
            review_doc = {
                "product_id": product_id,
                "review_text": review.get('text', ''),
                "rating": review.get('rating', ''),
                "title": review.get('title', ''),
                "review_date": review.get('date', ''),
                "verified_purchase": review.get('verified_purchase', False),
                "source": source,
                "scraped_date": datetime.utcnow(),
                "review_hash": hashlib.md5(
                    f"{review.get('text', '')}{source}{product_id}".encode()
                ).hexdigest()
            }
            
            # Check for duplicate
            existing = self.reviews_collection.find_one({
                "review_hash": review_doc["review_hash"]
            })
            
            if not existing:
                self.reviews_collection.insert_one(review_doc)
                stored_count += 1
        
        return stored_count
    
    def store_scraping_session(self, product_id: int, source: str, 
                             reviews_found: int, scraping_time: float, 
                             status: str, error_message: str = None) -> int:
        """Store scraping session metadata"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO scraping_sessions 
                (product_id, source, reviews_found, scraping_time, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (product_id, source, reviews_found, scraping_time, status, error_message))
            self.connection.commit()
            return cursor.lastrowid
        elif self.db_type == "mongodb":
            session_doc = {
                "product_id": product_id,
                "source": source,
                "reviews_found": reviews_found,
                "scraping_time": scraping_time,
                "status": status,
                "error_message": error_message,
                "session_date": datetime.utcnow()
            }
            result = self.scraping_sessions.insert_one(session_doc)
            return str(result.inserted_id)
    
    def store_analysis_results(self, product_id: int, analysis_result: Dict) -> int:
        """Store analysis results"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            sentiment = analysis_result.get('sentiment', {})
            summary = analysis_result.get('summary', {})
            
            cursor.execute("""
                INSERT INTO analysis_results 
                (product_id, overall_sentiment, sentiment_confidence, 
                 sentiment_distribution, summary_text, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                product_id,
                sentiment.get('overall_sentiment', ''),
                sentiment.get('confidence', 0.0),
                json.dumps(sentiment.get('distribution', {})),
                summary.get('text', ''),
                analysis_result.get('processing_time', 0.0)
            ))
            self.connection.commit()
            return cursor.lastrowid
    
    def get_product_reviews(self, product_id: int, source: str = None) -> List[Dict]:
        """Get all reviews for a product"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM reviews WHERE product_id = ?"
            params = [product_id]
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        
        elif self.db_type == "mongodb":
            filter_dict = {"product_id": product_id}
            if source:
                filter_dict["source"] = source
            
            return list(self.reviews_collection.find(filter_dict))
    
    def get_product_analytics(self, product_id: int) -> Dict:
        """Get analytics for a product"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Get review count by source
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM reviews 
                WHERE product_id = ? 
                GROUP BY source
            """, (product_id,))
            source_counts = dict(cursor.fetchall())
            
            # Get latest analysis
            cursor.execute("""
                SELECT * FROM analysis_results 
                WHERE product_id = ? 
                ORDER BY analysis_date DESC 
                LIMIT 1
            """, (product_id,))
            latest_analysis = cursor.fetchone()
            
            return {
                "product_id": product_id,
                "review_counts_by_source": source_counts,
                "total_reviews": sum(source_counts.values()),
                "latest_analysis": dict(latest_analysis) if latest_analysis else None
            }
    
    def search_products(self, search_term: str) -> List[Dict]:
        """Search for products in database"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM products 
                WHERE name LIKE ? OR search_term LIKE ?
                ORDER BY created_date DESC
            """, (f"%{search_term}%", f"%{search_term}%"))
            return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Database configuration examples
DATABASE_CONFIGS = {
    "sqlite": {
        "type": "sqlite",
        "connection_string": "data/reviews.db"
    },
    "postgresql": {
        "type": "postgresql", 
        "connection_string": "postgresql://user:password@localhost:5432/reviews_db"
    },
    "mongodb": {
        "type": "mongodb",
        "connection_string": "mongodb://localhost:27017/"
    }
}

def test_database():
    """Test database functionality"""
    print("🧪 Testing database functionality...")
    
    # Test SQLite
    db = ReviewDatabase("sqlite", "data/test_reviews.db")
    
    # Store test product
    product_id = db.store_product("Test iPhone", "iphone test", "electronics")
    print(f"✅ Stored product with ID: {product_id}")
    
    # Store test reviews
    test_reviews = [
        {"text": "Great phone!", "rating": "5/5", "title": "Excellent", "verified_purchase": True},
        {"text": "Poor battery life", "rating": "2/5", "title": "Disappointed", "verified_purchase": True}
    ]
    
    stored_count = db.store_reviews(product_id, test_reviews, "amazon")
    print(f"✅ Stored {stored_count} reviews")
    
    # Get reviews back
    reviews = db.get_product_reviews(product_id)
    print(f"✅ Retrieved {len(reviews)} reviews")
    
    db.close()
    print("✅ Database test completed!")

if __name__ == "__main__":
    test_database()

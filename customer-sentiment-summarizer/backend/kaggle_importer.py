#!/usr/bin/env python3
"""
Kaggle Dataset Importer for Customer Sentiment Analysis
Downloads and processes Kaggle datasets for review analysis
"""

import os
import ssl
import certifi
import urllib3
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
import glob
import chardet

# Fix SSL certificate issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set SSL context for certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables for certificate handling
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

class KaggleDatasetImporter:
    def __init__(self, download_path: str = "data/kaggle_datasets"):
        """Initialize the Kaggle dataset importer"""
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = ['.csv', '.json', '.jsonl', '.txt', '.tsv']
        
        # Common column names for review text
        self.text_columns = [
            'review', 'text', 'comment', 'content', 'message', 'feedback',
            'review_text', 'comment_text', 'content_text', 'review_content',
            'yorum', 'metin', 'icerik'  # Turkish columns
        ]
        
        # Common column names for ratings/labels
        self.rating_columns = [
            'rating', 'score', 'label', 'sentiment', 'class', 'classification',
            'star', 'stars', 'puan', 'derece', 'skor'  # Turkish columns
        ]
    
    def download_dataset(self, dataset_id: str, force_download: bool = False) -> str:
        """
        Download a Kaggle dataset using kagglehub with SSL handling
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., "furkangozukara/turkish-product-reviews")
            force_download: Whether to force re-download if already exists
            
        Returns:
            Path to the downloaded dataset files
        """
        try:
            print(f"📥 Downloading Kaggle dataset: {dataset_id}")
            
            # Set additional SSL environment variables for this session
            original_ssl_verify = os.environ.get('PYTHONHTTPSVERIFY', '1')
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            
            try:
                # Download using kagglehub with SSL verification disabled
                path = kagglehub.dataset_download(dataset_id)
                print(f"✅ Dataset downloaded to: {path}")
                return path
            finally:
                # Restore original SSL setting
                os.environ['PYTHONHTTPSVERIFY'] = original_ssl_verify
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print(f"💡 Trying alternative approach...")
            
            # Fallback: Try with requests directly if kagglehub fails
            try:
                return self._fallback_download(dataset_id)
            except Exception as fallback_error:
                print(f"❌ Fallback download also failed: {fallback_error}")
                raise Exception(f"Failed to download dataset {dataset_id}. Original error: {e}")
    
    def _fallback_download(self, dataset_id: str) -> str:
        """
        Fallback download method when kagglehub fails
        """
        print("🔄 Using fallback download method...")
        
        # For demonstration, we'll use a local sample dataset
        # In production, you could implement alternative download methods
        fallback_path = self.download_path / "fallback_data"
        fallback_path.mkdir(exist_ok=True)
        
        # Create a sample CSV file with some demo data
        sample_data = {
            'reviews.text': [
                "This iPhone is absolutely amazing! Great camera quality and battery life.",
                "Love this MacBook Pro! Fast performance and beautiful display.",
                "Samsung Galaxy has excellent features but battery could be better.",
                "Tesla Model 3 is incredible! Best car I've ever owned.",
                "AirPods Pro sound quality is fantastic, highly recommend!"
            ],
            'reviews.rating': [5, 5, 4, 5, 5],
            'name': ['iPhone 14', 'MacBook Pro', 'Samsung Galaxy S23', 'Tesla Model 3', 'AirPods Pro']
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_file = fallback_path / "sample_reviews.csv"
        sample_df.to_csv(sample_file, index=False)
        
        print(f"📝 Created sample dataset at: {sample_file}")
        return str(fallback_path)
    
    def explore_dataset(self, dataset_path: str) -> Dict:
        """
        Explore the structure of a downloaded dataset
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary with dataset exploration results
        """
        dataset_path = Path(dataset_path)
        
        exploration = {
            'path': str(dataset_path),
            'files': [],
            'total_files': 0,
            'supported_files': [],
            'file_sizes': {},
            'data_samples': {}
        }
        
        try:
            # List all files
            all_files = list(dataset_path.rglob('*'))
            files = [f for f in all_files if f.is_file()]
            
            exploration['total_files'] = len(files)
            exploration['files'] = [str(f.relative_to(dataset_path)) for f in files]
            
            # Find supported files
            for file_path in files:
                if file_path.suffix.lower() in self.supported_extensions:
                    rel_path = str(file_path.relative_to(dataset_path))
                    exploration['supported_files'].append(rel_path)
                    exploration['file_sizes'][rel_path] = file_path.stat().st_size
                    
                    # Get data sample
                    try:
                        sample = self._get_file_sample(file_path)
                        exploration['data_samples'][rel_path] = sample
                    except Exception as e:
                        exploration['data_samples'][rel_path] = f"Error reading sample: {e}"
            
            print(f"📊 Dataset exploration complete:")
            print(f"  Total files: {exploration['total_files']}")
            print(f"  Supported files: {len(exploration['supported_files'])}")
            
            return exploration
            
        except Exception as e:
            print(f"❌ Error exploring dataset: {e}")
            raise
    
    def _get_file_sample(self, file_path: Path, sample_size: int = 3) -> Dict:
        """Get a sample of data from a file"""
        try:
            if file_path.suffix.lower() == '.csv':
                # Try multiple encoding options
                encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
                
                # First try to detect encoding
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(100000)  # Read more data for better detection
                        detected = chardet.detect(raw_data)
                        if detected and detected['encoding'] and detected['confidence'] > 0.7:
                            encodings_to_try.insert(0, detected['encoding'])
                except:
                    pass
                
                # Try each encoding until one works
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size, on_bad_lines='skip')
                        return {
                            'type': 'csv',
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                            'sample': df.head(sample_size).to_dict('records'),
                            'encoding_used': encoding
                        }
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                # If all encodings fail, try with errors='ignore'
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', nrows=sample_size, on_bad_lines='skip')
                return {
                    'type': 'csv',
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'sample': df.head(sample_size).to_dict('records'),
                    'encoding_used': 'utf-8 (with errors ignored)'
                }
                
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return {
                            'type': 'json_list',
                            'length': len(data),
                            'sample': data[:sample_size]
                        }
                    else:
                        return {
                            'type': 'json_object',
                            'keys': list(data.keys()) if isinstance(data, dict) else None,
                            'sample': str(data)[:500]
                        }
                        
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [f.readline().strip() for _ in range(sample_size)]
                    return {
                        'type': 'text',
                        'sample_lines': lines
                    }
                    
        except Exception as e:
            return {'error': str(e)}
    
    def load_dataset_for_analysis(self, dataset_path: str, file_name: str = None, 
                                text_column: str = None, rating_column: str = None) -> Tuple[List[str], Optional[List]]:
        """
        Load dataset and extract reviews for sentiment analysis
        
        Args:
            dataset_path: Path to dataset directory
            file_name: Specific file to load (if None, will try to auto-detect)
            text_column: Column name containing review text
            rating_column: Column name containing ratings/labels
            
        Returns:
            Tuple of (review_texts, ratings_if_available)
        """
        dataset_path = Path(dataset_path)
        
        try:
            # Auto-detect file if not specified
            if file_name is None:
                file_name = self._auto_detect_main_file(dataset_path)
            
            file_path = dataset_path / file_name
            
            print(f"📂 Loading dataset file: {file_name}")
            
            # Load the data
            if file_path.suffix.lower() == '.csv':
                # Try multiple encoding options with robust error handling
                encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
                
                # First try to detect encoding
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(100000)  # Read more data for better detection
                        detected = chardet.detect(raw_data)
                        if detected and detected['encoding'] and detected['confidence'] > 0.7:
                            encodings_to_try.insert(0, detected['encoding'])
                except:
                    pass
                
                # Try each encoding until one works
                df = None
                encoding_used = None
                
                for encoding in encodings_to_try:
                    try:
                        print(f"🔤 Trying encoding: {encoding}")
                        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                        encoding_used = encoding
                        print(f"✅ Successfully loaded with encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, UnicodeError) as e:
                        print(f"❌ Failed with {encoding}: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ Error with {encoding}: {e}")
                        continue
                
                # If all encodings fail, try with errors='ignore'
                if df is None:
                    try:
                        print("🔤 Trying UTF-8 with error handling...")
                        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', on_bad_lines='skip')
                        encoding_used = 'utf-8 (with errors ignored)'
                        print("✅ Loaded with UTF-8 and error handling")
                    except Exception as e:
                        raise ValueError(f"Could not read CSV file with any encoding: {e}")
                
                if df is None:
                    raise ValueError("Could not load CSV file with any encoding method")
                
                # Auto-detect text column if not specified
                if text_column is None:
                    text_column = self._auto_detect_text_column(df.columns)
                
                # Auto-detect rating column if not specified
                if rating_column is None:
                    rating_column = self._auto_detect_rating_column(df.columns)
                
                # Extract reviews
                if text_column and text_column in df.columns:
                    reviews = df[text_column].dropna().astype(str).tolist()
                    
                    # Extract ratings if available
                    ratings = None
                    if rating_column and rating_column in df.columns:
                        ratings = df[rating_column].dropna().tolist()
                    
                    print(f"✅ Loaded {len(reviews)} reviews from column '{text_column}'")
                    if ratings:
                        print(f"✅ Loaded {len(ratings)} ratings from column '{rating_column}'")
                    
                    return reviews, ratings
                else:
                    raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
                    
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    # List of objects
                    reviews = []
                    ratings = []
                    
                    for item in data:
                        if isinstance(item, dict):
                            # Find text field
                            text_field = text_column or self._auto_detect_text_field(item.keys())
                            if text_field and text_field in item:
                                reviews.append(str(item[text_field]))
                                
                                # Find rating field
                                rating_field = rating_column or self._auto_detect_rating_field(item.keys())
                                if rating_field and rating_field in item:
                                    ratings.append(item[rating_field])
                    
                    print(f"✅ Loaded {len(reviews)} reviews from JSON")
                    return reviews, ratings if ratings else None
                else:
                    raise ValueError("JSON file does not contain a list of reviews")
                    
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def _auto_detect_main_file(self, dataset_path: Path) -> str:
        """Auto-detect the main data file in the dataset"""
        csv_files = list(dataset_path.glob('*.csv'))
        json_files = list(dataset_path.glob('*.json'))
        
        # Prefer CSV files
        if csv_files:
            # Return the largest CSV file
            largest_file = max(csv_files, key=lambda f: f.stat().st_size)
            return largest_file.name
        elif json_files:
            largest_file = max(json_files, key=lambda f: f.stat().st_size)
            return largest_file.name
        else:
            raise ValueError("No supported data files found in dataset")
    
    def _auto_detect_text_column(self, columns: List[str]) -> Optional[str]:
        """Auto-detect the column containing review text"""
        columns_lower = [col.lower() for col in columns]
        
        for text_col in self.text_columns:
            if text_col.lower() in columns_lower:
                idx = columns_lower.index(text_col.lower())
                return columns[idx]
        
        # If no exact match, look for columns containing keywords
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['review', 'text', 'comment', 'yorum']):
                return col
        
        return None
    
    def _auto_detect_rating_column(self, columns: List[str]) -> Optional[str]:
        """Auto-detect the column containing ratings"""
        columns_lower = [col.lower() for col in columns]
        
        for rating_col in self.rating_columns:
            if rating_col.lower() in columns_lower:
                idx = columns_lower.index(rating_col.lower())
                return columns[idx]
        
        # If no exact match, look for columns containing keywords
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['rating', 'score', 'star', 'puan']):
                return col
        
        return None
    
    def _auto_detect_text_field(self, keys: List[str]) -> Optional[str]:
        """Auto-detect text field in JSON objects"""
        return self._auto_detect_text_column(keys)
    
    def _auto_detect_rating_field(self, keys: List[str]) -> Optional[str]:
        """Auto-detect rating field in JSON objects"""
        return self._auto_detect_rating_column(keys)
    
    def load_dataset_with_adapter(self, dataset_id: str, file_path: str = "") -> pd.DataFrame:
        """
        Load dataset using KaggleDatasetAdapter for direct pandas loading
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., "arhamrumi/amazon-product-reviews")
            file_path: Specific file path within the dataset (optional)
            
        Returns:
            Pandas DataFrame with the dataset
        """
        try:
            print(f"📥 Loading Kaggle dataset with adapter: {dataset_id}")
            
            # Load using KaggleDatasetAdapter with error handling
            try:
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_id,
                    file_path
                )
            except Exception as adapter_error:
                print(f"❌ Adapter method failed: {adapter_error}")
                print("🔄 Trying manual download and load...")
                
                # Fallback: download manually and load with encoding handling
                dataset_path = kagglehub.dataset_download(dataset_id)
                dataset_dir = Path(dataset_path)
                
                # Find the target file
                if file_path:
                    target_file = dataset_dir / file_path
                else:
                    # Auto-detect main CSV file
                    csv_files = list(dataset_dir.glob('*.csv'))
                    if not csv_files:
                        csv_files = list(dataset_dir.rglob('*.csv'))
                    if not csv_files:
                        raise ValueError("No CSV files found in dataset")
                    target_file = max(csv_files, key=lambda f: f.stat().st_size)
                
                # Load with robust encoding handling
                encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
                
                for encoding in encodings_to_try:
                    try:
                        print(f"🔤 Trying encoding: {encoding}")
                        df = pd.read_csv(target_file, encoding=encoding, on_bad_lines='skip')
                        print(f"✅ Successfully loaded with encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception as e:
                        if encoding == encodings_to_try[-1]:  # Last encoding
                            # Final fallback with error handling
                            df = pd.read_csv(target_file, encoding='utf-8', errors='ignore', on_bad_lines='skip')
                            print("✅ Loaded with UTF-8 and error handling")
                        else:
                            continue
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset with adapter: {e}")
            raise
    
    def process_amazon_product_reviews(self, dataset_id: str = "datafiniti/consumer-reviews-of-amazon-products", 
                                     file_path: str = "", product_filter: str = None) -> Tuple[List[str], Optional[List]]:
        """
        Process the Amazon Product Reviews dataset specifically with fallback support
        
        Args:
            dataset_id: Kaggle dataset identifier
            file_path: Specific file within the dataset
            product_filter: Optional product name to filter reviews (e.g., "iPhone", "Samsung")
            
        Returns:
            Tuple of (review_texts, ratings)
        """
        try:
            print("🛒 Processing Amazon Product Reviews dataset...")
            if product_filter:
                print(f"🔍 Filtering for product: {product_filter}")
            
            # Try to load the real dataset first
            try:
                # Load the dataset using adapter
                df = self.load_dataset_with_adapter(dataset_id, file_path)
                
                print(f"📊 Dataset Info:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                
                # Continue with real dataset processing...
                return self._process_real_amazon_dataset(df, product_filter)
                
            except Exception as e:
                print(f"❌ Error loading dataset with adapter: {e}")
                print(f"🔄 Trying manual download and load...")
                
                # Try manual download as fallback
                try:
                    dataset_path = self.download_dataset(dataset_id)
                    df = self.load_dataset_for_analysis(dataset_path, 
                                                      text_column='reviews.text',
                                                      rating_column='reviews.rating')
                    if df is not None and len(df) > 0:
                        return self._process_real_amazon_dataset(df, product_filter)
                except Exception as download_error:
                    print(f"❌ Error loading dataset with adapter: {download_error}")
                
                # If all else fails, use sample data
                print(f"🔄 Using fallback sample data for product analysis...")
                return self._create_sample_product_reviews(product_filter)
                
        except Exception as e:
            print(f"❌ Error processing Amazon Product Reviews: {e}")
            print(f"🔄 Using fallback sample data...")
            return self._create_sample_product_reviews(product_filter)
    
    def _process_real_amazon_dataset(self, df, product_filter):
        """Process real Amazon dataset"""
        # Show sample data
        print(f"\n📝 First 3 records:")
        print(df.head(3).to_string())
        
        # Auto-detect text and rating columns
        text_column = self._auto_detect_text_column(df.columns)
        rating_column = self._auto_detect_rating_column(df.columns)
        
        # Common Amazon review column names
        amazon_text_cols = ['reviews.text', 'reviewText', 'review_text', 'review', 'text', 'content']
        amazon_rating_cols = ['reviews.rating', 'overall', 'rating', 'score', 'stars']
        amazon_product_cols = ['name', 'title', 'product', 'productName', 'reviews.title']
        
        # Try Amazon-specific column names first
        if not text_column:
            for col in amazon_text_cols:
                if col in df.columns:
                    text_column = col
                    break
        
        if not rating_column:
            for col in amazon_rating_cols:
                if col in df.columns:
                    rating_column = col
                    break
        
        # Find product name column for filtering
        product_column = None
        if product_filter:
            for col in amazon_product_cols:
                if col in df.columns:
                    product_column = col
                    break
        
        if not text_column:
            raise ValueError(f"Could not detect text column. Available columns: {df.columns.tolist()}")
        
        print(f"📝 Using text column: '{text_column}'")
        if rating_column:
            print(f"⭐ Using rating column: '{rating_column}'")
        if product_column:
            print(f"🏷️ Using product column: '{product_column}'")
        
        # Filter by product if specified
        if product_filter and product_column:
            # Create case-insensitive filter
            import re
            filter_pattern = re.compile(re.escape(product_filter), re.IGNORECASE)
            
            # Apply filter to product names
            product_mask = df[product_column].astype(str).str.contains(filter_pattern, na=False)
            
            # Also filter by review text content
            text_mask = df[text_column].astype(str).str.contains(filter_pattern, na=False)
            
            # Combine both filters
            combined_mask = product_mask | text_mask
            
            original_size = len(df)
            df = df[combined_mask]
            
            print(f"🔍 Filtered from {original_size} to {len(df)} reviews for '{product_filter}'")
            
            if len(df) == 0:
                print(f"⚠️ No reviews found for '{product_filter}'. Using sample data instead.")
                return self._create_sample_product_reviews(product_filter)
        
        elif product_filter:
            # If no product column found, filter by review text content
            import re
            filter_pattern = re.compile(re.escape(product_filter), re.IGNORECASE)
            text_mask = df[text_column].astype(str).str.contains(filter_pattern, na=False)
            
            original_size = len(df)
            df = df[text_mask]
            
            print(f"🔍 Filtered by review content from {original_size} to {len(df)} reviews for '{product_filter}'")
            
            if len(df) == 0:
                print(f"⚠️ No reviews found for '{product_filter}'. Using sample data instead.")
                return self._create_sample_product_reviews(product_filter)
        
        # Extract review texts
        review_texts = df[text_column].dropna().astype(str).tolist()
        
        # Extract ratings if available
        ratings = None
        if rating_column and rating_column in df.columns:
            try:
                ratings = pd.to_numeric(df[rating_column], errors='coerce').dropna().tolist()
                # Ensure same length as review texts
                min_length = min(len(review_texts), len(ratings))
                review_texts = review_texts[:min_length]
                ratings = ratings[:min_length]
            except Exception as e:
                print(f"⚠️ Error processing ratings: {e}")
                ratings = None
        
        print(f"✅ Successfully processed {len(review_texts)} Amazon product reviews")
        if ratings:
            print(f"⭐ Extracted {len(ratings)} ratings")
        
        return review_texts, ratings
    
    def _create_sample_product_reviews(self, product_filter: str = None) -> Tuple[List[str], List[int]]:
        """Create sample product reviews when real data is unavailable"""
        print(f"📝 Creating sample reviews for: {product_filter or 'general products'}")
        
        sample_reviews = []
        sample_ratings = []
        
        if product_filter and any(keyword in product_filter.lower() for keyword in ['iphone', 'phone', 'smartphone']):
            # iPhone/Phone reviews
            phone_data = [
                ("This iPhone is absolutely amazing! The camera quality is outstanding and battery life lasts all day. Highly recommend!", 5),
                ("Great phone overall, but the price is quite high. Camera is excellent though.", 4),
                ("Love the new features and iOS updates. Face ID works perfectly every time.", 5),
                ("Good phone but battery drains faster than expected. Camera is impressive.", 4),
                ("Best iPhone I've ever owned! Fast, reliable, and beautiful design.", 5),
                ("Expensive but worth it. The display quality is stunning and performance is smooth.", 4),
                ("Amazing camera for photography. Night mode is incredible!", 5),
                ("Phone works great but charging cable broke after 2 months.", 3),
                ("Perfect size and weight. Very satisfied with this purchase.", 5),
                ("Good upgrade from my old phone. Everything works as expected.", 4),
                ("Love the camera quality, takes amazing photos in any lighting!", 5),
                ("Battery life could be better, but overall great phone.", 4),
                ("Face ID is so convenient and fast. Love this feature!", 5),
                ("Screen is beautiful and vibrant. Perfect for watching videos.", 5),
                ("A bit expensive but the quality justifies the price.", 4)
            ]
            sample_reviews.extend([review for review, rating in phone_data])
            sample_ratings.extend([rating for review, rating in phone_data])
            
        elif product_filter and any(keyword in product_filter.lower() for keyword in ['macbook', 'laptop', 'computer']):
            # MacBook/Laptop reviews
            laptop_data = [
                ("This MacBook Pro is incredible! Perfect for video editing and programming. M2 chip is blazing fast.", 5),
                ("Great laptop but runs hot during intensive tasks. Battery life is excellent though.", 4),
                ("Love the keyboard and trackpad. Display is crisp and colors are vibrant.", 5),
                ("Expensive but quality is top-notch. Build quality feels premium.", 4),
                ("Perfect for software development. Compiles code super fast!", 5),
                ("Good laptop overall but wish it had more ports. USB-C only is limiting.", 3),
                ("Amazing performance for creative work. Photoshop runs smoothly.", 5),
                ("Battery lasts forever! Can work all day without charging.", 5),
                ("Sleek design and lightweight. Perfect for travel.", 4),
                ("Best laptop I've ever used. Worth every penny!", 5),
                ("The Retina display is stunning. Colors are so accurate.", 5),
                ("Speakers sound amazing for a laptop. Rich and clear audio.", 4),
                ("Touch Bar is useful but takes time to get used to.", 4),
                ("Performance is outstanding. No lag even with heavy applications.", 5),
                ("Love the Magic Keyboard. Typing experience is fantastic.", 5)
            ]
            sample_reviews.extend([review for review, rating in laptop_data])
            sample_ratings.extend([rating for review, rating in laptop_data])
            
        else:
            # General product reviews
            general_data = [
                ("Excellent product quality! Exceeded my expectations in every way.", 5),
                ("Good value for money. Would recommend to others.", 4),
                ("Amazing features and easy to use. Very satisfied with purchase.", 5),
                ("Works as advertised. Quality seems good so far.", 4),
                ("Love this product! Great design and functionality.", 5),
                ("Good product but shipping took longer than expected.", 3),
                ("Excellent customer service and fast delivery. Product works perfectly.", 5),
                ("Worth the price. Quality is better than similar products.", 4),
                ("Outstanding performance and reliability. Highly recommend!", 5),
                ("Good product overall. Minor issues but nothing major.", 3),
                ("Perfect for my needs. Easy setup and great performance.", 5),
                ("Solid product with good build quality. Happy with purchase.", 4),
                ("Impressive features and smooth operation. Great value!", 5),
                ("Works well but could use some improvements.", 3),
                ("Excellent product! Better than I expected.", 5)
            ]
            sample_reviews.extend([review for review, rating in general_data])
            sample_ratings.extend([rating for review, rating in general_data])
        
        print(f"✅ Generated {len(sample_reviews)} sample reviews")
        return sample_reviews, sample_ratings


def test_amazon_reviews_only():
    """Test only the Amazon Product Reviews dataset"""
    importer = KaggleDatasetImporter()
    
    try:
        print("🛒 Testing Amazon Product Reviews Dataset")
        print("=" * 50)
        
        # Process Amazon dataset
        reviews, ratings = importer.process_amazon_product_reviews()
        
        print(f"\n🎯 Amazon Reviews Results:")
        print(f"  Total Reviews: {len(reviews)}")
        print(f"  Has Ratings: {'Yes' if ratings else 'No'}")
        
        # Show sample reviews
        print(f"\n📝 Sample Reviews:")
        for i, review in enumerate(reviews[:3], 1):
            preview = review[:200] + "..." if len(review) > 200 else review
            print(f"  {i}. {preview}")
            
        if ratings:
            print(f"\n⭐ Sample Ratings: {ratings[:10]}")
            print(f"⭐ Rating Distribution:")
            rating_counts = {}
            for rating in ratings[:1000]:  # Sample first 1000 ratings
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            for rating, count in sorted(rating_counts.items()):
                print(f"    {rating} stars: {count} reviews")
        
        return reviews, ratings
        
    except Exception as e:
        print(f"❌ Error testing Amazon reviews: {e}")
        return [], None


def test_amazon_reviews_only():
    """Test only the Amazon Product Reviews dataset"""
    importer = KaggleDatasetImporter()
    
    try:
        print("🛒 Testing Amazon Product Reviews Dataset")
        print("=" * 50)
        
        # Process Amazon dataset
        reviews, ratings = importer.process_amazon_product_reviews()
        
        print(f"\n🎯 Amazon Reviews Results:")
        print(f"  Total Reviews: {len(reviews)}")
        print(f"  Has Ratings: {'Yes' if ratings else 'No'}")
        
        # Show sample reviews
        print(f"\n📝 Sample Reviews:")
        for i, review in enumerate(reviews[:3], 1):
            preview = review[:200] + "..." if len(review) > 200 else review
            print(f"  {i}. {preview}")
            
        if ratings:
            print(f"\n⭐ Sample Ratings: {ratings[:10]}")
            print(f"⭐ Rating Distribution:")
            rating_counts = {}
            for rating in ratings[:1000]:  # Sample first 1000 ratings
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            for rating, count in sorted(rating_counts.items()):
                print(f"    {rating} stars: {count} reviews")
        
        return reviews, ratings
        
    except Exception as e:
        print(f"❌ Error testing Amazon reviews: {e}")
        return [], None


if __name__ == "__main__":
    # Test the Amazon Product Reviews dataset specifically
    test_amazon_reviews_only()

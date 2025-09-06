"""
Bulk Review Import Module
Supports importing reviews from various formats (CSV, JSON, TXT)
"""

import csv
import json
import pandas as pd
from typing import List, Dict, Any
import re

class BulkReviewImporter:
    def __init__(self):
        """Initialize the bulk review importer"""
        self.supported_formats = ['csv', 'json', 'txt', 'xlsx']
    
    def import_from_csv(self, file_path: str, text_column: str = 'review_text') -> List[str]:
        """
        Import reviews from CSV file
        
        Args:
            file_path: Path to CSV file
            text_column: Name of column containing review text
            
        Returns:
            List of review texts
        """
        reviews = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if text_column in row and row[text_column].strip():
                        reviews.append(row[text_column].strip())
            print(f"✅ Imported {len(reviews)} reviews from CSV")
        except Exception as e:
            print(f"❌ Error importing CSV: {e}")
        
        return reviews
    
    def import_from_json(self, file_path: str, text_field: str = 'text') -> List[str]:
        """
        Import reviews from JSON file
        
        Args:
            file_path: Path to JSON file
            text_field: Field name containing review text
            
        Returns:
            List of review texts
        """
        reviews = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and text_field in item:
                            reviews.append(item[text_field].strip())
                        elif isinstance(item, str):
                            reviews.append(item.strip())
                elif isinstance(data, dict) and 'reviews' in data:
                    for review in data['reviews']:
                        if text_field in review:
                            reviews.append(review[text_field].strip())
            
            print(f"✅ Imported {len(reviews)} reviews from JSON")
        except Exception as e:
            print(f"❌ Error importing JSON: {e}")
        
        return reviews
    
    def import_from_txt(self, file_path: str, separator: str = '\n\n') -> List[str]:
        """
        Import reviews from text file
        
        Args:
            file_path: Path to text file
            separator: Separator between reviews
            
        Returns:
            List of review texts
        """
        reviews = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                raw_reviews = content.split(separator)
                
                for review in raw_reviews:
                    cleaned = review.strip()
                    if len(cleaned) > 20:  # Filter out very short text
                        reviews.append(cleaned)
            
            print(f"✅ Imported {len(reviews)} reviews from text file")
        except Exception as e:
            print(f"❌ Error importing text file: {e}")
        
        return reviews
    
    def import_from_excel(self, file_path: str, text_column: str = 'review_text') -> List[str]:
        """
        Import reviews from Excel file
        
        Args:
            file_path: Path to Excel file
            text_column: Name of column containing review text
            
        Returns:
            List of review texts
        """
        reviews = []
        try:
            df = pd.read_excel(file_path)
            if text_column in df.columns:
                reviews = df[text_column].dropna().astype(str).tolist()
                reviews = [r.strip() for r in reviews if len(r.strip()) > 20]
            
            print(f"✅ Imported {len(reviews)} reviews from Excel")
        except Exception as e:
            print(f"❌ Error importing Excel: {e}")
        
        return reviews
    
    def clean_reviews(self, reviews: List[str]) -> List[str]:
        """
        Clean and filter reviews
        
        Args:
            reviews: List of raw review texts
            
        Returns:
            List of cleaned review texts
        """
        cleaned = []
        
        for review in reviews:
            # Remove extra whitespace
            review = re.sub(r'\s+', ' ', review.strip())
            
            # Remove common unwanted patterns
            review = re.sub(r'Read more.*$', '', review)
            review = re.sub(r'Helpful.*$', '', review)
            review = re.sub(r'\d+\s*out of\s*\d+\s*found this helpful', '', review)
            
            # Filter minimum length
            if len(review) > 30:
                cleaned.append(review)
        
        return cleaned
    
    def auto_detect_format(self, file_path: str) -> str:
        """Auto-detect file format based on extension"""
        extension = file_path.lower().split('.')[-1]
        if extension in self.supported_formats:
            return extension
        return 'txt'
    
    def import_reviews(self, file_path: str, format_type: str = None, **kwargs) -> List[str]:
        """
        Import reviews from file with auto-format detection
        
        Args:
            file_path: Path to file
            format_type: File format (auto-detected if None)
            **kwargs: Additional arguments for specific importers
            
        Returns:
            List of review texts
        """
        if not format_type:
            format_type = self.auto_detect_format(file_path)
        
        reviews = []
        
        if format_type == 'csv':
            reviews = self.import_from_csv(file_path, **kwargs)
        elif format_type == 'json':
            reviews = self.import_from_json(file_path, **kwargs)
        elif format_type == 'txt':
            reviews = self.import_from_txt(file_path, **kwargs)
        elif format_type in ['xlsx', 'xls']:
            reviews = self.import_from_excel(file_path, **kwargs)
        else:
            print(f"❌ Unsupported format: {format_type}")
            return []
        
        # Clean reviews
        cleaned_reviews = self.clean_reviews(reviews)
        
        print(f"🧹 Cleaned {len(reviews)} → {len(cleaned_reviews)} reviews")
        
        return cleaned_reviews

def create_sample_files():
    """Create sample import files for testing"""
    
    # Sample reviews
    sample_reviews = [
        "Amazing product! The quality exceeded my expectations. Fast delivery and great customer service.",
        "Decent product but overpriced. Could be better for the money. Shipping was slow.",
        "Love this! Perfect for my needs. Great build quality and excellent performance.",
        "Terrible experience. Product broke after one week. Would not recommend.",
        "Good value for money. Works as expected. No complaints so far.",
        "Outstanding! This is exactly what I was looking for. Highly recommended!",
        "Average product. Nothing special but does the job. Okay for the price.",
        "Disappointed with this purchase. Quality is poor and doesn't match description."
    ]
    
    # Create CSV sample
    with open('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['review_text', 'rating', 'product'])
        for i, review in enumerate(sample_reviews):
            writer.writerow([review, f"{(i % 5) + 1}/5", "Sample Product"])
    
    # Create JSON sample
    json_data = {
        "product": "Sample Product",
        "reviews": [
            {"text": review, "rating": (i % 5) + 1} 
            for i, review in enumerate(sample_reviews)
        ]
    }
    
    with open('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.json', 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)
    
    # Create TXT sample
    with open('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.txt', 'w', encoding='utf-8') as file:
        file.write('\n\n'.join(sample_reviews))
    
    print("✅ Created sample import files:")
    print("  - sample_reviews.csv")
    print("  - sample_reviews.json") 
    print("  - sample_reviews.txt")

if __name__ == "__main__":
    # Create sample files
    create_sample_files()
    
    # Test importer
    importer = BulkReviewImporter()
    
    # Test CSV import
    reviews_csv = importer.import_reviews('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.csv', 
                                         text_column='review_text')
    print(f"CSV: {len(reviews_csv)} reviews")
    
    # Test JSON import
    reviews_json = importer.import_reviews('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.json',
                                          text_field='text')
    print(f"JSON: {len(reviews_json)} reviews")
    
    # Test TXT import
    reviews_txt = importer.import_reviews('/Users/sarthakray/Sentiment-Tracker-Final/customer-sentiment-summarizer/sample_reviews.txt')
    print(f"TXT: {len(reviews_txt)} reviews")

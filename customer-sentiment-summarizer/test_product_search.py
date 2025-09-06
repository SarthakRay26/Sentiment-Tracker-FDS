#!/usr/bin/env python3
"""
Test Product Search Functionality
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

def map_product_to_dataset(product_query: str) -> dict:
    """Map product search queries to appropriate datasets"""
    
    # Define dataset mappings
    datasets = [
        {
            'dataset_id': 'datafiniti/consumer-reviews-of-amazon-products',
            'description': 'Amazon Product Reviews (28K+ reviews)',
            'keywords': ['amazon', 'product', 'general', 'electronics', 'phone', 'iphone', 'samsung', 'laptop', 'computer', 'tablet', 'ipad', 'headphones', 'speaker', 'camera', 'tv', 'appliance'],
            'filter': None  # No filtering needed for general Amazon products
        },
        {
            'dataset_id': 'snap/amazon-fine-food-reviews',
            'description': 'Amazon Fine Food Reviews',
            'keywords': ['food', 'snack', 'coffee', 'tea', 'drink', 'beverage', 'nutrition', 'supplement', 'kitchen', 'cooking'],
            'filter': None
        }
    ]
    
    # Find the best matching dataset
    for dataset in datasets:
        if any(keyword in product_query.lower() for keyword in dataset['keywords']):
            return dataset
    
    # Default to Amazon products dataset
    return datasets[0]

def test_product_search():
    """Test the product search functionality"""
    
    test_queries = [
        "iPhone 15 Pro Max",
        "Samsung Galaxy",
        "MacBook Pro",
        "coffee",
        "food supplements",
        "random product name"
    ]
    
    print("🔍 Testing Product Search Mapping...\n")
    
    for query in test_queries:
        mapping = map_product_to_dataset(query)
        print(f"Query: '{query}'")
        print(f"  → Dataset: {mapping['dataset_id']}")
        print(f"  → Description: {mapping['description']}")
        print(f"  → Filter: {mapping.get('filter', 'None')}")
        print()

def test_full_analysis(product_query="Amazon products", sample_size=5):
    """Test the full analysis pipeline"""
    
    print(f"🔍 Testing full analysis for: {product_query}")
    
    try:
        # Map product to dataset
        dataset_mapping = map_product_to_dataset(product_query)
        print(f"📦 Using dataset: {dataset_mapping['dataset_id']}")
        
        # Import and use the kaggle importer
        from kaggle_importer import KaggleDatasetImporter
        
        # Initialize importer
        importer = KaggleDatasetImporter()
        
        # Load the dataset
        dataset_path = importer.download_dataset(dataset_mapping['dataset_id'])
        reviews, ratings = importer.load_dataset_for_analysis(
            dataset_path,
            text_column='reviews.text',
            rating_column='reviews.rating'
        )
        
        if not reviews:
            print("❌ No reviews found")
            return
        
        # Sample the data
        if len(reviews) > sample_size:
            import random
            indices = random.sample(range(len(reviews)), sample_size)
            reviews = [reviews[i] for i in indices]
            if ratings:
                ratings = [ratings[i] for i in indices if i < len(ratings)]
        
        print(f"✅ Found {len(reviews)} reviews")
        
        # Show sample review
        if reviews:
            print(f"📝 Sample review: {reviews[0][:100]}...")
        
        # Analyze sentiment
        from main_simple import CustomerSentimentSummarizer
        analyzer = CustomerSentimentSummarizer()
        
        result = analyzer.analyze_customer_feedback(reviews)
        
        if result.get('success'):
            sentiment = result.get('sentiment', {})
            print(f"✅ Analysis successful!")
            print(f"📊 Overall sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
            print(f"🎯 Confidence: {sentiment.get('confidence', 0):.1%}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Product Search Test\n")
    
    # Test mapping function
    test_product_search()
    
    # Test full pipeline
    print("="*50)
    test_full_analysis("iPhone", 5)

"""
Demo script to test the complete scraping and analysis system
"""

import sys
import os
sys.path.append('backend')

from backend.main_simple import CustomerSentimentSummarizer

def create_demo_reviews(product_name):
    """Create demo reviews for testing"""
    demo_reviews = [
        f"I absolutely love this {product_name}! The battery life is incredible and easily lasts all day. The camera quality is outstanding, especially the night mode. Fast delivery and excellent packaging. Highly recommend!",
        
        f"Terrible experience with this {product_name}. It broke after just two weeks of normal use. Customer service was unhelpful and refused to provide a refund. Complete waste of money.",
        
        f"Good value for money {product_name}. The battery life is decent but not exceptional. The screen quality is nice and bright. Delivery was on time. Overall satisfied but nothing extraordinary.",
        
        f"Amazing camera quality on this {product_name}! The photos are crisp and colors are vibrant. Battery life could be better though - barely lasts a full day with heavy usage. Price is reasonable for the features.",
        
        f"Poor build quality {product_name}. The product feels cheap and flimsy. Battery drains very quickly even with minimal use. Would not recommend to anyone. Save your money and buy something else.",
        
        f"Excellent {product_name} overall. Fast performance and smooth operation. The display is beautiful and very responsive. Battery life is good but charging could be faster. Great customer service experience.",
        
        f"Mixed feelings about this {product_name} purchase. Love the design and build quality, but the battery performance is disappointing. Camera is decent in good lighting but struggles in low light.",
        
        f"Outstanding value {product_name}! This product exceeded my expectations in every way. Fast performance, long battery life, excellent camera, and beautiful design. Customer service was helpful and responsive.",
    ]
    
    return demo_reviews

def test_complete_system():
    """Test the complete system with demo data"""
    print("🎯 Testing Complete Customer Sentiment Summarizer System\n")
    
    try:
        # Initialize the system
        print("🚀 Initializing system...")
        analyzer = CustomerSentimentSummarizer()
        
        # Show system status
        status = analyzer.get_system_status()
        print("\n🔧 System Status:")
        for component, available in status.items():
            if component != 'timestamp':
                emoji = '✅' if available else '❌'
                print(f"  {component}: {emoji}")
        
        # Test with demo product
        product_name = "iPhone 15 Pro"
        print(f"\n📱 Testing with product: {product_name}")
        
        # Create demo reviews
        demo_reviews = create_demo_reviews(product_name)
        print(f"📝 Created {len(demo_reviews)} demo reviews")
        
        # Analyze the demo reviews
        print("\n🧠 Analyzing demo reviews...")
        result = analyzer.analyze_customer_feedback(
            demo_reviews,
            include_topics=True,
            include_summary=True
        )
        
        # Display results
        print("\n📊 ANALYSIS RESULTS:")
        print("=" * 50)
        
        # Sentiment Results
        sentiment = result.get('sentiment', {})
        print(f"🎯 Overall Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        print(f"🎯 Confidence: {sentiment.get('confidence', 0):.1%}")
        
        if sentiment.get('distribution'):
            print("\n📈 Sentiment Distribution:")
            for emotion, percentage in sentiment['distribution'].items():
                print(f"  {emotion.title()}: {percentage:.1%}")
        
        # Summary Results
        summary = result.get('summary', {})
        if summary.get('success'):
            print(f"\n📝 Generated Summary:")
            print(f"  {summary.get('text', 'No summary available')}")
            print(f"  Compression: {summary.get('compression_ratio', 0):.1%}")
        
        # Processing Info
        print(f"\n⏱️ Processing Time: {result.get('processing_time', 0)} seconds")
        print(f"🔧 Components Used: {', '.join(result.get('components_used', []))}")
        
        # Insights
        insights = result.get('insights', {})
        if insights:
            print("\n💡 Insights:")
            for key, value in insights.items():
                if isinstance(value, str):
                    print(f"  {key}: {value}")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scraping_simulation():
    """Simulate the scraping workflow"""
    print("\n🕷️ SIMULATING WEB SCRAPING WORKFLOW")
    print("=" * 50)
    
    try:
        analyzer = CustomerSentimentSummarizer()
        
        # Simulate scraping result
        product_name = "Samsung Galaxy S24"
        demo_reviews = create_demo_reviews(product_name)
        
        print(f"🔍 [SIMULATED] Searching for: {product_name}")
        print(f"🕷️ [SIMULATED] Scraping reviews from Amazon...")
        print(f"✅ [SIMULATED] Found {len(demo_reviews)} reviews")
        
        # Analyze the simulated scraped reviews
        print("🧠 Analyzing scraped reviews...")
        start_time = analyzer.analyze_customer_feedback.__globals__['time'].time()
        result = analyzer.analyze_customer_feedback(demo_reviews)
        analysis_time = analyzer.analyze_customer_feedback.__globals__['time'].time() - start_time
        
        # Create simulated scraping result
        simulated_result = {
            'product_name': product_name,
            'platform': 'amazon',
            'scraping_time': 2.5,  # Simulated scraping time
            'analysis_time': analysis_time,
            'total_time': 2.5 + analysis_time,
            'reviews_found': len(demo_reviews),
            'scraping_details': {
                'sources_attempted': ['amazon'],
                'sources_successful': ['amazon'],
                'total_reviews': len(demo_reviews)
            },
            'analysis': result,
            'sample_reviews': demo_reviews[:3],
            'success': True
        }
        
        # Display scraping simulation results
        print(f"\n📊 SCRAPING & ANALYSIS RESULTS:")
        print(f"🛒 Platform: {simulated_result['platform'].title()}")
        print(f"📝 Reviews Found: {simulated_result['reviews_found']}")
        print(f"🕷️ Scraping Time: {simulated_result['scraping_time']} seconds")
        print(f"🧠 Analysis Time: {simulated_result['analysis_time']:.2f} seconds")
        print(f"⏱️ Total Time: {simulated_result['total_time']:.2f} seconds")
        
        analysis = simulated_result['analysis']
        sentiment = analysis.get('sentiment', {})
        print(f"\n🎯 Overall Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        
        summary = analysis.get('summary', {})
        if summary.get('success'):
            print(f"\n📝 Summary: {summary.get('text', 'No summary')[:150]}...")
        
        print("\n📝 Sample Reviews:")
        for i, review in enumerate(simulated_result['sample_reviews'][:2], 1):
            print(f"{i}. {review[:100]}...")
        
        print("\n✅ Scraping simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Scraping simulation failed: {e}")
        return False

if __name__ == "__main__":
    # Test complete system
    success1 = test_complete_system()
    
    # Test scraping simulation
    success2 = test_scraping_simulation()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! The system is ready for use.")
        print("\n🚀 To start the web interface, run:")
        print("   python frontend/app.py")
        print("\n💡 Features available:")
        print("   ✅ Sentiment Analysis")
        print("   ✅ Text Summarization") 
        print("   ✅ Web Scraping (simulated)")
        print("   ✅ Interactive Gradio Interface")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

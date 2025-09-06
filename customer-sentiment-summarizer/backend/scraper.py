"""
Web Scraper Module for E-commerce Reviews
Automatically fetch customer reviews from various e-commerce platforms
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings("ignore")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available. Some scraping features will be limited.")

class ReviewScraper:
    def __init__(self):
        """Initialize the review scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Add some basic session properties to mimic real browser behavior
        self.session.cookies.set('session-token', 'mock-session-token')
        self.session.cookies.set('ubid-main', 'mock-ubid-main')
    
    def setup_driver(self) -> Optional[webdriver.Chrome]:
        """Setup Chrome driver with stealth options"""
        if not SELENIUM_AVAILABLE:
            return None
        
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            return None
    
    def scrape_amazon_reviews(self, product_name: str, max_reviews: int = 50) -> Dict:
        """
        Scrape Amazon reviews for a specific product
        
        Args:
            product_name: Name of the product to search for
            max_reviews: Maximum number of reviews to fetch
            
        Returns:
            Dictionary containing reviews and metadata
        """
        print(f"🔍 Searching Amazon for: {product_name}")
        
        try:
            # Search for the product
            search_query = quote_plus(product_name)
            search_url = f"https://www.amazon.com/s?k={search_query}&ref=sr_pg_1"
            
            # Add random delay to appear more human-like
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(search_url)
            if response.status_code != 200:
                return {"error": f"Failed to search Amazon: {response.status_code}", "success": False}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find product links
            product_containers = soup.find_all('div', {'data-component-type': 's-search-result'})
            if not product_containers:
                return {"error": "No products found on Amazon", "success": False}
            
            # Get the first product's ASIN
            first_product = product_containers[0]
            asin = first_product.get('data-asin')
            if not asin:
                return {"error": "Could not find product ASIN", "success": False}
            
            print(f"📦 Found product ASIN: {asin}")
            
            # Try multiple approaches to get reviews
            reviews = []
            
            # Approach 1: Try to get reviews from product page first
            try:
                reviews.extend(self._scrape_reviews_from_product_page(asin, max_reviews // 2))
                print(f"📄 Got {len(reviews)} reviews from product page")
            except Exception as e:
                print(f"⚠️ Product page scraping failed: {e}")
            
            # Approach 2: Try dedicated reviews page if we need more reviews
            if len(reviews) < max_reviews:
                remaining_reviews = max_reviews - len(reviews)
                try:
                    reviews_page_reviews = self._scrape_reviews_from_reviews_page(asin, remaining_reviews)
                    reviews.extend(reviews_page_reviews)
                    print(f"📄 Got {len(reviews_page_reviews)} additional reviews from reviews page")
                except Exception as e:
                    print(f"⚠️ Reviews page scraping failed: {e}")
            
            # If still no reviews, return a helpful response
            if not reviews:
                return {
                    "error": "Amazon is currently blocking automated access. This is a common protection mechanism. You can manually copy and paste reviews using the 'Manual Analysis' tab.",
                    "success": False,
                    "suggestion": "Try using the manual review analysis feature instead"
                }
            
            print(f"✅ Successfully scraped {len(reviews)} Amazon reviews")
            return {
                "reviews": reviews,
                "total": len(reviews),
                "source": "Amazon",
                "product_name": product_name,
                "success": True
            }
        
        except Exception as e:
            return {
                "error": f"Error scraping Amazon: {str(e)}",
                "success": False,
                "suggestion": "Try using the manual review analysis feature instead"
            }
    
    def _scrape_reviews_from_product_page(self, asin: str, max_reviews: int) -> List[Dict]:
        """Try to scrape reviews directly from the product page"""
        product_url = f"https://www.amazon.com/dp/{asin}"
        
        time.sleep(random.uniform(1, 2))
        response = self.session.get(product_url)
        
        if response.status_code != 200:
            raise Exception(f"Product page returned status {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = []
        
        # Look for review widgets on product page
        review_elements = soup.find_all('div', {'data-hook': re.compile('review|cr-')})
        if not review_elements:
            # Try alternative selectors
            review_elements = soup.find_all('div', class_=re.compile('review'))
        
        for review_elem in review_elements[:max_reviews]:
            try:
                review_data = self._extract_review_data(review_elem, "Amazon Product Page")
                if review_data and review_data.get('text'):
                    reviews.append(review_data)
            except Exception as e:
                continue
        
        return reviews
    
    def _scrape_reviews_from_reviews_page(self, asin: str, max_reviews: int) -> List[Dict]:
        """Try to scrape from dedicated reviews page with enhanced techniques"""
        reviews = []
        page = 1
        max_pages = min(3, (max_reviews // 10) + 1)
        
        while len(reviews) < max_reviews and page <= max_pages:
            print(f"📄 Trying Amazon reviews page {page}")
            
            # Try different URL formats
            urls_to_try = [
                f"https://www.amazon.com/product-reviews/{asin}?pageNumber={page}",
                f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?pageNumber={page}",
                f"https://www.amazon.com/reviews/{asin}?pageNumber={page}"
            ]
            
            page_reviews = []
            for url in urls_to_try:
                try:
                    time.sleep(random.uniform(2, 4))  # Longer delay
                    response = self.session.get(url)
                    
                    if response.status_code == 200 and 'robot' not in response.text.lower():
                        soup = BeautifulSoup(response.content, 'html.parser')
                        review_elements = soup.find_all('div', {'data-hook': 'review'})
                        
                        if review_elements:
                            for review_elem in review_elements:
                                if len(page_reviews) >= 10:  # Limit per page
                                    break
                                
                                try:
                                    review_data = self._extract_review_data(review_elem, "Amazon Reviews Page")
                                    if review_data and review_data.get('text'):
                                        page_reviews.append(review_data)
                                except Exception as e:
                                    continue
                            break  # Success, don't try other URLs
                    
                except Exception as e:
                    continue
            
            reviews.extend(page_reviews)
            if not page_reviews:  # No reviews found, stop trying
                break
                
            page += 1
        
        return reviews
    
    def _extract_review_data(self, review_elem, source_page: str) -> Dict:
        """Extract review data from a review element"""
        # Extract rating
        rating_elem = review_elem.find('i', {'data-hook': 'review-star-rating'})
        rating = "N/A"
        if rating_elem:
            rating_text = rating_elem.get('class', [''])[0]
            rating_match = re.search(r'star-(\d)', rating_text)
            if rating_match:
                rating = f"{rating_match.group(1)}/5"
        
        # Extract title
        title_elem = review_elem.find('a', {'data-hook': 'review-title'})
        if not title_elem:
            title_elem = review_elem.find('span', {'data-hook': 'review-title'})
        title = title_elem.get_text(strip=True) if title_elem else "No title"
        
        # Extract review text
        text_elem = review_elem.find('span', {'data-hook': 'review-body'})
        text = text_elem.get_text(strip=True) if text_elem else ""
        
        # Extract date
        date_elem = review_elem.find('span', {'data-hook': 'review-date'})
        date = date_elem.get_text(strip=True) if date_elem else "No date"
        
        # Extract verified purchase
        verified_elem = review_elem.find('span', {'data-hook': 'avp-badge'})
        verified = bool(verified_elem)
        
        # Only return if we have substantial text
        if text and len(text.strip()) > 20:
            return {
                'rating': rating,
                'title': title,
                'text': text.strip(),
                'date': date,
                'verified_purchase': verified,
                'source': source_page,
                'product_name': ""  # Will be filled by caller
            }
        
        return None
    
    def scrape_flipkart_reviews(self, product_name: str, max_reviews: int = 50) -> Dict:
        """
        Scrape Flipkart reviews for a specific product
        
        Args:
            product_name: Name of the product to search for
            max_reviews: Maximum number of reviews to fetch
            
        Returns:
            Dictionary containing reviews and metadata
        """
        print(f"🔍 Searching Flipkart for: {product_name}")
        
        try:
            # Search for the product
            search_query = quote_plus(product_name)
            search_url = f"https://www.flipkart.com/search?q={search_query}"
            
            response = self.session.get(search_url)
            if response.status_code != 200:
                return {"error": f"Failed to search Flipkart: {response.status_code}"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find product links (Flipkart structure)
            product_links = soup.find_all('a', class_=re.compile('_1fQZEK|s1Q9rs'))
            if not product_links:
                # Try alternative selectors
                product_links = soup.find_all('a', href=re.compile('/p/'))
            
            if not product_links:
                return {"error": "No products found on Flipkart"}
            
            # Get first product URL
            product_url = "https://www.flipkart.com" + product_links[0]['href']
            print(f"📦 Found product URL: {product_url}")
            
            # Get product page
            response = self.session.get(product_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for reviews section or rating summary
            reviews = []
            
            # Try to find review elements on the product page
            review_elements = soup.find_all('div', class_=re.compile('_16PBlm|_27M-vq'))
            
            for review_elem in review_elements[:max_reviews]:
                try:
                    # Extract rating
                    rating_elem = review_elem.find('div', class_=re.compile('_3LWZlK|hGSR34'))
                    rating = rating_elem.get_text(strip=True) if rating_elem else "N/A"
                    
                    # Extract review text
                    text_elem = review_elem.find('div', class_=re.compile('_1AtVbE|t-ZTKy'))
                    if not text_elem:
                        text_elem = review_elem.find('div', class_='_1AtVbE')
                    
                    text = text_elem.get_text(strip=True) if text_elem else ""
                    
                    if text and len(text) > 20:
                        reviews.append({
                            'rating': rating,
                            'title': "Flipkart Review",
                            'text': text,
                            'date': "N/A",
                            'verified_purchase': True,
                            'source': 'Flipkart',
                            'product_name': product_name
                        })
                
                except Exception as e:
                    continue
            
            print(f"✅ Successfully scraped {len(reviews)} Flipkart reviews")
            return {
                "reviews": reviews,
                "total": len(reviews),
                "source": "Flipkart",
                "product_name": product_name,
                "success": True
            }
        
        except Exception as e:
            return {"error": f"Error scraping Flipkart: {str(e)}", "success": False}
    
    def scrape_multiple_sources(self, product_name: str, max_reviews_per_source: int = 30) -> Dict:
        """
        Scrape reviews from multiple e-commerce sources
        
        Args:
            product_name: Name of the product to search for
            max_reviews_per_source: Maximum reviews per platform
            
        Returns:
            Combined results from all sources
        """
        print(f"🌐 Scraping multiple sources for: {product_name}")
        
        all_reviews = []
        sources_attempted = []
        sources_successful = []
        errors = []
        suggestions = []
        
        # Scrape Amazon
        sources_attempted.append("Amazon")
        amazon_result = self.scrape_amazon_reviews(product_name, max_reviews_per_source)
        if amazon_result.get("success"):
            all_reviews.extend(amazon_result["reviews"])
            sources_successful.append("Amazon")
            # Update product_name for all reviews
            for review in amazon_result["reviews"]:
                review["product_name"] = product_name
        else:
            error_msg = amazon_result.get('error', 'Unknown error')
            errors.append(f"Amazon: {error_msg}")
            if amazon_result.get('suggestion'):
                suggestions.append(amazon_result['suggestion'])
        
        time.sleep(2)  # Delay between sources
        
        # Scrape Flipkart
        sources_attempted.append("Flipkart")
        flipkart_result = self.scrape_flipkart_reviews(product_name, max_reviews_per_source)
        if flipkart_result.get("success"):
            all_reviews.extend(flipkart_result["reviews"])
            sources_successful.append("Flipkart")
            # Update product_name for all reviews
            for review in flipkart_result["reviews"]:
                review["product_name"] = product_name
        else:
            error_msg = flipkart_result.get('error', 'Unknown error')
            errors.append(f"Flipkart: {error_msg}")
            if flipkart_result.get('suggestion'):
                suggestions.append(flipkart_result['suggestion'])
        
        # Prepare final result
        success = len(all_reviews) > 0
        final_result = {
            "reviews": all_reviews,
            "total": len(all_reviews),
            "product_name": product_name,
            "sources_attempted": sources_attempted,
            "sources_successful": sources_successful,
            "errors": errors,
            "success": success
        }
        
        # Add helpful message if no reviews found
        if not success:
            if any("blocking" in error.lower() or "robot" in error.lower() for error in errors):
                final_result["user_message"] = (
                    "🤖 **Scraping Currently Blocked**\n\n"
                    "E-commerce sites are actively blocking automated access. This is normal behavior to protect their servers.\n\n"
                    "**Alternative Options:**\n"
                    "1. **Manual Analysis**: Copy and paste reviews directly using the 'Single Review' or 'Multiple Reviews' tabs\n"
                    "2. **Try Later**: Anti-bot measures may be temporary\n"
                    "3. **Different Product**: Some products may have less protection\n\n"
                    "The manual analysis features work perfectly and provide the same insights!"
                )
            else:
                final_result["user_message"] = (
                    "❌ **No Reviews Found**\n\n"
                    "We couldn't find reviews for this product on the attempted platforms.\n\n"
                    "**Suggestions:**\n"
                    "• Try a different product name or variation\n"
                    "• Use the manual analysis tabs to copy/paste reviews directly\n"
                    "• Check if the product exists on Amazon or Flipkart"
                )
        
        return final_result
    
    def get_review_texts_only(self, scraping_result: Dict) -> List[str]:
        """
        Extract only the review texts from scraping result
        
        Args:
            scraping_result: Result from scraping methods
            
        Returns:
            List of review text strings
        """
        if not scraping_result.get("success") or not scraping_result.get("reviews"):
            return []
        
        return [review["text"] for review in scraping_result["reviews"] if review.get("text")]
    
    def save_reviews_to_file(self, scraping_result: Dict, filename: str):
        """Save scraped reviews to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(scraping_result, f, indent=2, ensure_ascii=False)
            print(f"💾 Reviews saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving reviews: {e}")

def test_scraper():
    """Test the review scraper"""
    scraper = ReviewScraper()
    
    # Test product
    product_name = "iPhone 15"
    print(f"🧪 Testing scraper with product: {product_name}")
    
    # Test Amazon scraping
    print("\n📱 Testing Amazon scraping...")
    amazon_result = scraper.scrape_amazon_reviews(product_name, max_reviews=10)
    
    if amazon_result.get("success"):
        print(f"✅ Amazon: Found {amazon_result['total']} reviews")
        if amazon_result["reviews"]:
            print(f"Sample review: {amazon_result['reviews'][0]['text'][:100]}...")
    else:
        print(f"❌ Amazon failed: {amazon_result.get('error')}")
    
    # Test multiple sources
    print("\n🌐 Testing multiple sources...")
    multi_result = scraper.scrape_multiple_sources(product_name, max_reviews_per_source=5)
    
    print(f"Sources attempted: {multi_result['sources_attempted']}")
    print(f"Sources successful: {multi_result['sources_successful']}")
    print(f"Total reviews found: {multi_result['total']}")
    
    if multi_result.get("errors"):
        print(f"Errors encountered: {multi_result['errors']}")
    
    return multi_result

if __name__ == "__main__":
    test_scraper()

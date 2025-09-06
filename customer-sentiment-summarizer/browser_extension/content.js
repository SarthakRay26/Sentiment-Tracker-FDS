// Content script to extract reviews from e-commerce pages
(function() {
    'use strict';

    // Amazon review selectors
    const AMAZON_SELECTORS = {
        reviews: '[data-hook="review"]',
        reviewText: '[data-hook="review-body"] span',
        rating: '[data-hook="review-star-rating"]',
        title: '[data-hook="review-title"] span'
    };

    // Flipkart review selectors
    const FLIPKART_SELECTORS = {
        reviews: '._27M-vq, ._1AtVbE',
        reviewText: '._1AtVbE div',
        rating: '.hGSR34, ._3LWZlK',
        title: '.t-ZTKy'
    };

    function detectSite() {
        const hostname = window.location.hostname;
        if (hostname.includes('amazon')) return 'amazon';
        if (hostname.includes('flipkart')) return 'flipkart';
        return 'unknown';
    }

    function extractAmazonReviews() {
        const reviews = [];
        const reviewElements = document.querySelectorAll(AMAZON_SELECTORS.reviews);
        
        reviewElements.forEach(element => {
            try {
                const textElement = element.querySelector(AMAZON_SELECTORS.reviewText);
                const ratingElement = element.querySelector(AMAZON_SELECTORS.rating);
                const titleElement = element.querySelector(AMAZON_SELECTORS.title);
                
                const text = textElement ? textElement.textContent.trim() : '';
                const rating = ratingElement ? ratingElement.getAttribute('class') : '';
                const title = titleElement ? titleElement.textContent.trim() : '';
                
                if (text && text.length > 20) {
                    reviews.push({
                        text: text,
                        title: title,
                        rating: rating,
                        source: 'Amazon'
                    });
                }
            } catch (e) {
                console.log('Error extracting review:', e);
            }
        });
        
        return reviews;
    }

    function extractFlipkartReviews() {
        const reviews = [];
        const reviewElements = document.querySelectorAll(FLIPKART_SELECTORS.reviews);
        
        reviewElements.forEach(element => {
            try {
                const text = element.textContent.trim();
                
                if (text && text.length > 20) {
                    reviews.push({
                        text: text,
                        title: 'Flipkart Review',
                        rating: 'N/A',
                        source: 'Flipkart'
                    });
                }
            } catch (e) {
                console.log('Error extracting review:', e);
            }
        });
        
        return reviews;
    }

    function extractReviews() {
        const site = detectSite();
        let reviews = [];
        
        switch (site) {
            case 'amazon':
                reviews = extractAmazonReviews();
                break;
            case 'flipkart':
                reviews = extractFlipkartReviews();
                break;
            default:
                // Generic extraction for other sites
                const allText = document.querySelectorAll('p, div, span');
                allText.forEach(element => {
                    const text = element.textContent.trim();
                    if (text.length > 50 && text.length < 1000) {
                        // Simple heuristic: if it looks like a review
                        if (text.includes('product') || text.includes('quality') || 
                            text.includes('good') || text.includes('bad') ||
                            text.includes('recommend') || text.includes('love') ||
                            text.includes('hate')) {
                            reviews.push({
                                text: text,
                                title: 'Generic Review',
                                rating: 'N/A',
                                source: site
                            });
                        }
                    }
                });
                break;
        }
        
        return reviews;
    }

    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === 'extractReviews') {
            const reviews = extractReviews();
            sendResponse({
                success: true,
                reviews: reviews,
                count: reviews.length,
                site: detectSite()
            });
        }
    });

})();

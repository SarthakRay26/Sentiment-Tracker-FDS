document.addEventListener('DOMContentLoaded', function() {
    const extractBtn = document.getElementById('extractBtn');
    const copyBtn = document.getElementById('copyBtn');
    const openAnalyzer = document.getElementById('openAnalyzer');
    const resultDiv = document.getElementById('result');
    const reviewsTextDiv = document.getElementById('reviewsText');
    
    let extractedReviews = [];
    
    extractBtn.addEventListener('click', function() {
        extractBtn.textContent = '⏳ Extracting...';
        extractBtn.disabled = true;
        
        // Get the active tab
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            // Send message to content script
            chrome.tabs.sendMessage(tabs[0].id, { action: 'extractReviews' }, function(response) {
                extractBtn.textContent = '📥 Extract Reviews';
                extractBtn.disabled = false;
                
                if (chrome.runtime.lastError) {
                    showResult('❌ Error: Cannot extract from this page. Make sure you\'re on a product review page.', 'error');
                    return;
                }
                
                if (response && response.success) {
                    extractedReviews = response.reviews;
                    const reviewTexts = extractedReviews.map(review => review.text).join('\n\n');
                    
                    showResult(`✅ Found ${response.count} reviews from ${response.site}`, 'success');
                    reviewsTextDiv.textContent = reviewTexts;
                    reviewsTextDiv.style.display = 'block';
                    copyBtn.style.display = 'block';
                } else {
                    showResult('❌ No reviews found on this page', 'error');
                }
            });
        });
    });
    
    copyBtn.addEventListener('click', function() {
        const reviewTexts = extractedReviews.map(review => review.text).join('\n\n');
        
        navigator.clipboard.writeText(reviewTexts).then(function() {
            copyBtn.textContent = '✅ Copied!';
            setTimeout(() => {
                copyBtn.textContent = '📋 Copy to Clipboard';
            }, 2000);
        }).catch(function() {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = reviewTexts;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            copyBtn.textContent = '✅ Copied!';
            setTimeout(() => {
                copyBtn.textContent = '📋 Copy to Clipboard';
            }, 2000);
        });
    });
    
    openAnalyzer.addEventListener('click', function() {
        chrome.tabs.create({ url: 'http://localhost:7860' });
    });
    
    function showResult(message, type) {
        resultDiv.textContent = message;
        resultDiv.className = `result ${type}`;
        resultDiv.style.display = 'block';
    }
});

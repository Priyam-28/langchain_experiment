#!/usr/bin/env python3
"""
Demo script to test the product search functionality
"""

import os
from main import search_product, extract_page_content, analyze_with_gemini

# For demo purposes, you can set your API keys here
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
# os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

def demo_search():
    """Demo function to test product search"""
    product = "iPhone 15 Pro"
    
    print(f"Demo: Searching for {product}")
    
    # Test search
    results = search_product(product)
    print(f"Found {len(results)} results")
    
    if results:
        # Extract first 2 URLs for demo
        urls = [r.get('url', r) if isinstance(r, dict) else r for r in results[:2]]
        print(f"Extracting content from {len(urls)} URLs")
        
        # Test extraction
        extracted = extract_page_content(urls)
        print(f"Extracted content from {len(extracted)} pages")
        
        # Test analysis
        analysis = analyze_with_gemini(product, extracted)
        print("\nAnalysis:")
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)

if __name__ == "__main__":
    demo_search()

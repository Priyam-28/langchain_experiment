import getpass
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up API keys
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter API key for Tavily: ")

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.schema import HumanMessage

# Initialize models and tools
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
search_tool = TavilySearch(max_results=5)
extract_tool = TavilyExtract()

def search_product(product_name: str) -> List[Dict]:
    """Search for a product and return top 5 results with detailed comparison"""
    
    # Extract potential features from the product name
    features = []
    for keyword in ["noise cancellation", "anc", "wireless", "bluetooth", "waterproof", 
                   "sport", "gaming", "bass", "premium", "budget", "true wireless", "tws", "neckband"]:
        if keyword.lower() in product_name.lower():
            features.append(keyword)
    
    # Build a more targeted search query
    feature_string = " ".join(features)
    base_query = f"best {product_name} comparison"
    
    if features:
        search_query = f"{base_query} with {feature_string} detailed review price specifications"
    else:
        search_query = f"{base_query} top models detailed review price specifications"
    
    try:
        # Add search filters to prioritize comparison articles and reviews
        search_results = search_tool.invoke({
            "query": search_query,
            "search_depth": "advanced",
            "include_domains": ["amazon.com", "flipkart.com", "cnet.com", "rtings.com", 
                              "techradar.com", "theverge.com", "headphonezone.in", "boat-lifestyle.com"]
        })
        print(f"Found {len(search_results)} results for '{product_name}'")
        return search_results
    except Exception as e:
        print(f"Error during search: {e}")
        try:
            # Fallback to simpler search if advanced fails
            search_results = search_tool.invoke({"query": search_query})
            print(f"Fallback search found {len(search_results)} results")
            return search_results
        except Exception as e:
            print(f"Fallback search also failed: {e}")
            return []

def extract_page_content(urls: List[str]) -> List[Dict]:
    """Extract content from the top 5 URLs"""
    extracted_content = []
    
    for i, url in enumerate(urls[:5], 1):
        try:
            print(f"Extracting content from page {i}: {url}")
            content = extract_tool.invoke({"urls": [url]})
            extracted_content.append({
                "url": url,
                "content": content,
                "page_number": i
            })
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            extracted_content.append({
                "url": url,
                "content": f"Error extracting content: {e}",
                "page_number": i
            })
    
    return extracted_content

def analyze_with_gemini(product_name: str, extracted_data: List[Dict]) -> str:
    """Use Gemini to analyze the extracted content and provide insights"""
    
    # Prepare the content for analysis
    content_summary = f"Product Search Analysis for: {product_name}\n\n"
    
    for page_data in extracted_data:
        content_summary += f"Page {page_data['page_number']}: {page_data['url']}\n"
        content_summary += f"Content: {str(page_data['content'])[:1000]}...\n\n"
    
    # Extract key features from product name
    features = []
    for keyword in ["noise cancellation", "wireless", "bluetooth", "waterproof", "sport", 
                   "gaming", "bass", "premium", "budget", "true wireless", "neckband"]:
        if keyword.lower() in product_name.lower():
            features.append(keyword)
    
    prompt = f"""
    Analyze the following product search results for "{product_name}" and provide a detailed comparison:
    
    PART 1: COMPLETE PRODUCT LIST
    First, identify and list ALL unique product models found across all pages (at least 10 if available).
    For each product include:
    - Exact product name and model number
    - Price range (if mentioned)
    - Key specifications
    
    PART 2: FEATURE ANALYSIS
    Analyze these specific features across all products:
    - Sound quality and audio performance
    - Battery life
    - Comfort and design
    - Connectivity options
    - Water/dust resistance ratings
    - Special features (ANC, EQ customization, etc.)
    
    PART 3: FILTERED RECOMMENDATIONS
    Based on the user's interest in {product_name}{' with ' + ', '.join(features) if features else ''}, recommend:
    1. Best overall option
    2. Best budget option
    3. Best premium option
    4. Best for specific use cases (workout, travel, etc.)
    5. Best value for money
    
    PART 4: COMPARISON TABLE
    Create a comparison table of the top 5 most relevant options showing:
    - Product name
    - Price
    - Key features
    - Battery life
    - Pros and cons
    - Overall rating
    
    PART 5: BUYING ADVICE
    - Best places to buy with current prices
    - Any ongoing deals or promotions
    - Availability information
    - Any upcoming new models worth waiting for
    
    Search Results:
    {content_summary}
    
    Focus especially on extracting factual information and organizing it in a clear, structured format that helps the user make an informed decision.
    """
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error analyzing with Gemini: {e}"

def main():
    print("=== Product Search and Analysis Tool ===\n")
    
    # Get product name from user
    product_name = input("Enter the product you want to search for: ").strip()
    
    if not product_name:
        print("Please enter a valid product name.")
        return
    
    print(f"\n🔍 Searching for '{product_name}'...")
    
    # Step 1: Search for the product
    search_results = search_product(product_name)
    
    if not search_results:
        print("No search results found. Please try a different product name.")
        return
    
    # Extract URLs from search results
    urls = []
    print("\n📋 Search Results:")
    for i, result in enumerate(search_results, 1):
        if isinstance(result, dict) and 'url' in result:
            url = result['url']
            title = result.get('title', 'No title')
            print(f"{i}. {title}")
            print(f"   URL: {url}")
            urls.append(url)
        elif isinstance(result, str):
            # Handle case where result might be a URL string
            urls.append(result)
            print(f"{i}. {result}")
    
    if not urls:
        print("No valid URLs found in search results.")
        return
    
    print(f"\n📄 Extracting content from top {min(5, len(urls))} pages...")
    
    # Step 2: Extract content from top 5 pages
    extracted_data = extract_page_content(urls)
    
    print(f"\n🤖 Analyzing results with Gemini...")
    
    # Step 3: Analyze with Gemini
    analysis = analyze_with_gemini(product_name, extracted_data)
    
    print("\n" + "="*80)
    print(f"🎧 GEMINI ANALYSIS: {product_name.upper()} 🎧")
    print("="*80)
    
    # Format the analysis for better readability with section headers
    formatted_analysis = analysis
    # Bold section headers if terminal supports it
    for section in ["COMPLETE PRODUCT LIST", "FEATURE ANALYSIS", "FILTERED RECOMMENDATIONS", 
                    "COMPARISON TABLE", "BUYING ADVICE"]:
        if section in formatted_analysis:
            formatted_analysis = formatted_analysis.replace(
                section, 
                f"\033[1m{section}\033[0m"
            )
    
    print(formatted_analysis)
    print("="*80)
    
    # Optional: Save results to file
    save_results = input("\nWould you like to save the results to a file? (y/n): ").strip().lower()
    if save_results == 'y':
        filename = f"{product_name.replace(' ', '_')}_analysis.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Product Analysis for: {product_name}\n")
                f.write("="*50 + "\n\n")
                f.write("Search Results:\n")
                for i, result in enumerate(search_results, 1):
                    f.write(f"{i}. {result}\n")
                f.write(f"\n\nGemini Analysis:\n{analysis}\n")
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()



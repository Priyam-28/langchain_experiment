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
    """Search for a product and return top 5 results"""
    search_query = f"best {product_name} amazon,flipkart,myntra"
    
    try:
        search_results = search_tool.invoke({"query": search_query})
        print(f"Found {len(search_results)} results for '{product_name}'")
        return search_results
    except Exception as e:
        print(f"Error during search: {e}")
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
    
    prompt = f"""
    Analyze the following product search results for "{product_name}" and provide a comprehensive summary including:
    
    1. Product availability and pricing information
    2. Key features mentioned across the pages
    3. Customer reviews or ratings if available
    4. Best places to buy based on the search results
    5. Any notable deals or promotions
    
    Search Results:
    {content_summary}
    
    Please provide a well-structured analysis with clear recommendations.
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
    print(search_results)
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
    for i in urls:
        print(i)
    if not urls:
        print("No valid URLs found in search results.")
        return
    
    print(f"\n📄 Extracting content from top {min(5, len(urls))} pages...")
    
if __name__ == "__main__":
    main()
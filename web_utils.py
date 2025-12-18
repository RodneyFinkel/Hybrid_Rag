# utils/web_utils.py
import requests
from bs4 import BeautifulSoup
import re
import logging

DEFAULT_INSTRUCTIONS = "Extract the main readable text content from the page."
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' 
DEFAULT_TIMEOUT = 10

def browse_page(url: str, instructions: str = None, user_agent: str = DEFAULT_USER_AGENT, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch and extract clean text from a webpage."""
    if not instructions:
        instructions = DEFAULT_INSTRUCTIONS
    
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logging.warning(f"Browse failed for {url}: status {response.status_code}")
            return "Page unavailable"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()  # Remove more noise
        
        text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', text).strip()
        clean_text = clean_text[:3000]  # Slightly longer safe limit (~750 tokens)
        
        if len(clean_text) < 100:
            logging.warning(f"Insufficient content extracted from {url}")
            return "Insufficient content on page."
        
        logging.info(f"Successfully browsed {url}: {len(clean_text)} chars extracted")
        return clean_text
        
    except requests.RequestException as e:
        logging.error(f"Request error for {url}: {str(e)}")
        return "Page fetch failed."
    except Exception as e:
        logging.error(f"Unexpected error browsing {url}: {str(e)}")
        return "Page processing failed."
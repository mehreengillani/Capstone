# engines.py
#from openai import OpenAI
#from google import genai


# engines.py
from openai import OpenAI
from perplexity import Perplexity
import time
from openai import RateLimitError

# ============ API KEYS ============
OPENAI_API_KEY = "" # HIDDEN
PERPLEXITY_API_KEY = "" # HIDDEN


# ============ OPENAI ============
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def query_openai(prompt, max_retries=3):
    """Query OpenAI with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = openai_client.responses.create(
                model="gpt-5.6-luna",
                input=prompt,
                tools=[{"type": "web_search"}],
            )
            answer = response.output_text
            
            citations = []
            for item in response.output:
                if hasattr(item, 'content'):
                    for block in item.content:
                        if hasattr(block, 'annotations'):
                            for ann in block.annotations:
                                if ann.type == "url_citation":
                                    citations.append(ann.url)
            
            return answer, citations
        
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
            print(f"    [OpenAI rate limited. Waiting {wait}s...]")
            time.sleep(wait)
    
    return "", []


# ============ PERPLEXITY ============
perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)

def query_perplexity(prompt):
    """Query Perplexity using the new Agent API."""
    response = perplexity_client.responses.create(
        preset="fast",
        input=prompt,
    )
    
    answer = response.output_text
    
    # Extract citations
    citations = []
    for item in response.output:
        if hasattr(item, 'type') and item.type == "search_results":
            if hasattr(item, 'results'):
                for r in item.results:
                    if hasattr(r, 'url'):
                        citations.append(r.url)
    
    return answer, citations


# ============ GEMINI (disabled) ============
def query_gemini(prompt):
    """Gemini is temporarily disabled due to quota limits."""
    return "", []
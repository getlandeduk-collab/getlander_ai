from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
from typing import Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Assuming these utility functions exist in utils.py
# from .utils import extract_title, get_text_from_soup, clean_soup

# Mock utility functions (replace with your actual implementations)
def clean_soup(soup):
    """Remove script and style elements"""
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    return soup

def get_text_from_soup(soup):
    """Extract text content from soup"""
    text = soup.get_text(separator='\n', strip=True)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def extract_title(soup):
    """Extract title from soup"""
    if soup.title:
        return soup.title.string
    elif soup.find('h1'):
        return soup.find('h1').get_text(strip=True)
    return "No title found"

# BeautifulSoupScraper class
class BeautifulSoupScraper:
    def __init__(self, link, session=None):
        self.link = link
        self.session = session if session else requests.Session()

    def scrape(self):
        """
        Scrapes content from a webpage by making a GET request, parsing the HTML using
        BeautifulSoup, and extracting script and style elements before returning the cleaned content.
        """
        try:
            response = self.session.get(self.link, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=response.encoding
            )

            soup = clean_soup(soup)
            content = get_text_from_soup(soup)
            title = extract_title(soup)

            return content, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", ""

# FastAPI application
app = FastAPI(
    title="Web Scraper API",
    description="API for scraping web content using BeautifulSoup",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ScrapeRequest(BaseModel):
    url: HttpUrl
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com"
            }
        }

class ScrapeResponse(BaseModel):
    success: bool
    url: str
    title: str
    content: str
    content_length: int
    error: Optional[str] = None

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Web Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "POST /scrape": "Scrape a webpage",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape content from a given URL
    
    - **url**: The URL to scrape (must be a valid HTTP/HTTPS URL)
    
    Returns the scraped content, title, and metadata
    """
    try:
        # Create a session for the request
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create scraper and scrape the URL
        scraper = BeautifulSoupScraper(str(request.url), session=session)
        content, title = scraper.scrape()
        
        if not content and not title:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape the URL. The page may be inaccessible or invalid."
            )
        
        return ScrapeResponse(
            success=True,
            url=str(request.url),
            title=title,
            content=content,
            content_length=len(content)
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=408,
            detail="Request timeout while trying to access the URL"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to the URL"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
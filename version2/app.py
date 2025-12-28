from __future__ import annotations

import asyncio
import html
import json
import os
import time
import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging with environment variable control
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)



from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks

from fastapi.responses import JSONResponse, StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from pydantic import ValidationError

from dotenv import load_dotenv

from pathlib import Path



from models import (

    MatchJobsJsonRequest,

    MatchJobsRequest,

    MatchJobsResponse,

    CandidateProfile,

    JobPosting,

    MatchedJob,

    ProgressStatus,

    Settings,

    FirebaseResume,

    FirebaseResumeListResponse,

    FirebaseResumeResponse,

    SavedCVResponse,

    GetUserResumesRequest,

    GetUserResumeRequest,

    GetUserResumePdfRequest,

    GetUserResumeBase64Request,

    GetUserSavedCvsRequest,

    ExtractJobInfoRequest,

    JobInfoExtracted,

    PlaywrightScrapeResponse,

    SummarizeJobRequest,

    SummarizeJobResponse,

    SponsorshipInfo,
    ApolloPersonSearchRequest,
    ApolloPersonSearchResponse,
    ApolloEnrichPersonRequest,
    ApolloEnrichPersonResponse,
    SponsorshipCheckRequest,
)
from utils import (
    decode_base64_pdf,
    extract_text_from_pdf_bytes,
    now_iso,
    make_request_id,
    redact_long_text,
    scrape_website_custom,
    is_authorized_sponsor,
)
from agents import build_resume_parser, build_scraper, build_scorer, build_summarizer

from pyngrok import ngrok, conf as ngrok_conf



# Optional imports for HTML parsing

try:

    import requests

    from bs4 import BeautifulSoup

except ImportError:

    requests = None

    BeautifulSoup = None





# Load environment from root .env and version2/.env if present

load_dotenv()  # project root

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)



# CRITICAL: Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS is explicitly set from system environment

# This is needed because async context might not have access to system env vars

# Priority: GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string) > GOOGLE_APPLICATION_CREDENTIALS (file path)



# Check for JSON string first (preferred for production)

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in os.environ:

    # Try to get from system environment (Windows environment variables)

    import sys

    import subprocess

    try:

        # On Windows, try to get from system environment

        result = subprocess.run(

            ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS_JSON", "User")'],

            capture_output=True,

            text=True,

            timeout=2

        )

        if result.returncode == 0 and result.stdout.strip():

            os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = result.stdout.strip()

    except:

        pass  # Non-critical, continue anyway



# Fallback to file path method if JSON not found

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:

    # Try to get from system environment (Windows environment variables)

    import sys

    import subprocess

    try:

        # On Windows, try to get from system environment

        result = subprocess.run(

            ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "User")'],

            capture_output=True,

            text=True,

            timeout=2

        )

        if result.returncode == 0 and result.stdout.strip():

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = result.stdout.strip()

    except:

        pass  # Non-critical, continue anyway



app = FastAPI(title="Intelligent Job Matching API", version="0.1.0")



# Ngrok startup (optional)

NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

NGROK_DOMAIN = os.getenv("NGROK_DOMAIN") or os.getenv("NGROK_URL") or "gobbler-fresh-sole.ngrok-free.app"

if NGROK_AUTHTOKEN and not os.getenv("DISABLE_NGROK"):

    try:

        ngrok_conf.get_default().auth_token = NGROK_AUTHTOKEN

        # Ensure no old tunnels keep port busy

        for t in ngrok.get_tunnels():

            try:

                ngrok.disconnect(t.public_url)

            except Exception:

                pass

        if NGROK_DOMAIN:

            print(f"[NGROK] Connecting to domain: {NGROK_DOMAIN}")

            ngrok.connect(addr="8000", proto="http", domain=NGROK_DOMAIN)

            # Get the public URL

            tunnels = ngrok.get_tunnels()

            if tunnels:

                print(f"[NGROK] Public URL: {tunnels[0].public_url}")

        else:

            ngrok.connect(addr="8000", proto="http")

    except Exception as e:

        # Non-fatal if ngrok fails

        print(f"[NGROK] Error: {e}")

        pass



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)


# Startup event: Preload sponsorship CSV data
@app.on_event("startup")
async def startup_event():
    """Preload sponsorship CSV data at application startup for faster lookups."""
    try:
        from sponsorship_checker import load_sponsorship_data
        logger.info("Preloading sponsorship CSV data...")
        load_sponsorship_data()  # This will cache the data
        logger.info("Sponsorship CSV data loaded and cached successfully")
    except Exception as e:
        logger.warning(f"Failed to preload sponsorship CSV (non-fatal): {e}")
        # Non-fatal - will load on first use





# In-memory stores

REQUEST_PROGRESS: Dict[str, ProgressStatus] = {}

SCRAPE_CACHE: Dict[str, Dict[str, Any]] = {}

LAST_REQUESTS_BY_IP: Dict[str, List[float]] = {}





def get_settings() -> Settings:

    return Settings(

        openai_api_key=os.getenv("OPENAI_API_KEY"),

        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),

        model_name=os.getenv("OPENAI_MODEL", "gpt-5-mini"),

        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),

        max_concurrent_scrapes=int(os.getenv("MAX_CONCURRENT_SCRAPES", "8")),

        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),

        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),

    )





async def rate_limit(request: Request, settings: Settings = Depends(get_settings)):

    ip = request.client.host if request.client else "unknown"

    window = 60.0

    max_req = settings.rate_limit_requests_per_minute

    now = time.time()

    bucket = LAST_REQUESTS_BY_IP.setdefault(ip, [])

    # prune

    while bucket and now - bucket[0] > window:

        bucket.pop(0)

    if len(bucket) >= max_req:

        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    bucket.append(now)




async def stream_openai_response(prompt: str, model_name: str = "gpt-4o-mini", openai_api_key: Optional[str] = None):
    """
    Stream OpenAI API response as Server-Sent Events (SSE).
    
    Args:
        prompt: The prompt to send to OpenAI
        model_name: The model to use (default: gpt-4o-mini)
        openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
    
    Yields:
        SSE-formatted chunks of the response
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        
        # Note: Some models don't support temperature=0, so we omit it to use default
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                # Format as SSE
                yield f"data: {json.dumps({'content': content, 'type': 'token'})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        # Send error as SSE
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


def clean_job_title(title: Optional[str]) -> Optional[str]:
    """
    Clean and normalize job title, removing patterns like "job_title:**Name:M/L developer".
    
    Args:
        title: Raw job title string
        
    Returns:
        Cleaned job title or None if invalid
    """
    if not title or not isinstance(title, str):
        return None
    
    # Remove leading/trailing whitespace
    title = title.strip()
    
    # Remove patterns like "job_title:**", "job_title:", "Title:**", "Name:**", etc.
    title = re.sub(r'^(job_title|title|name|position|role)\s*[:\*]+\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^\*+\s*', '', title)  # Remove leading asterisks
    title = re.sub(r'\s*\*+\s*$', '', title)  # Remove trailing asterisks
    
    # Remove common prefixes/suffixes
    title = re.sub(r'^[^:]*:\s*', '', title)  # Remove "Job Board: " or "Category: "
    title = re.sub(r'\s*[-–—|]\s*at\s+[^-]+$', '', title, flags=re.I)  # Remove " - at Company Name"
    title = re.sub(r'\s*[-–—|]\s*[^-]+(?:\.com|\.in|\.org).*$', '', title, flags=re.I)  # Remove website suffixes
    # Only remove " - Company Name" if it looks like a company name (not part of job title like "Front-End")
    # Be very conservative - only remove if it's clearly a company suffix pattern
    # Pattern: " - CompanyName" or " - Company Name Ltd" at the end, but NOT if it contains job keywords
    job_keywords = ['developer', 'engineer', 'manager', 'analyst', 'specialist', 'architect', 'designer', 'scientist', 'consultant', 'coordinator', 'officer', 'executive', 'director', 'lead', 'senior', 'junior', 'front', 'back', 'end', 'full', 'stack', 'react', 'angular', 'vue', 'node', 'python', 'java', 'c++']
    # Check if title contains job keywords - if so, don't remove anything after dash (it's part of title)
    if not any(keyword in title.lower() for keyword in job_keywords):
        # Only remove company suffixes if no job keywords found
        # Match patterns like " - Robert Half" or " - Company Ltd" at the end
        title = re.sub(r'\s*[-–—|]\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Ltd|Limited|Inc|Corp|LLC|LLP|PLC))?)$', '', title)
    title = re.sub(r'\s*[|]\s*', ' ', title)  # Replace pipe separators with space
    
    # Remove quotes and special characters at start/end
    title = re.sub(r'^["\'\`]+|["\'\`]+$', '', title)
    
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title)
    title = title.strip()
    
    # Validate title quality
    if not title or len(title) < 3:
        return None
    
    if len(title) > 150:
        title = title[:150].strip()
    
    # Remove if it looks like navigation or invalid
    invalid_patterns = [
        r'^(home|menu|navigation|skip to|cookie|privacy policy)',
        r'^(not specified|unknown|n/a|na|none)$',
        r'^[\*\-\s]+$',  # Only asterisks, dashes, or spaces
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, title, re.IGNORECASE):
            return None
    
    return title


def clean_company_name(company: Optional[str]) -> Optional[str]:
    """
    Clean and normalize company name, removing patterns like "Name**: Company".
    
    Args:
        company: Raw company name string
        
    Returns:
        Cleaned company name or None if invalid
    """
    if not company or not isinstance(company, str):
        return None
    
    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <, &gt; -> >)
    company = html.unescape(company)
    
    # Remove leading/trailing whitespace
    company = company.strip()
    
    # Remove patterns like "Name**:", "Company:", "Employer:", etc.
    company = re.sub(r'^(Name\*{0,2}:?\s*|Company:?\s*|Employer:?\s*|Organization:?\s*)', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^\*+\s*', '', company)  # Remove leading asterisks
    company = re.sub(r'\s*\*+\s*$', '', company)  # Remove trailing asterisks
    
    # Remove common prefixes
    company = re.sub(r'^at\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^for\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^with\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^by\s+', '', company, flags=re.IGNORECASE)
    
    # Remove quotes and special characters
    company = re.sub(r'^["\'\`]+|["\'\`]+$', '', company)
    company = re.sub(r'^[\*\-\s]+|[\*\-\s]+$', '', company)
    
    # Remove truncated text indicators
    if company.endswith('...') or company.endswith('…'):
        return None  # Truncated company names are invalid
    
    # Remove if it ends mid-word (likely truncated)
    if len(company) > 50 and not company[-1].isalnum() and not company.endswith(('Ltd', 'Inc', 'LLC', 'Corp', 'Corporation', 'Group', 'Holdings')):
        # Likely truncated
        return None
    
    # Normalize whitespace
    company = re.sub(r'\s+', ' ', company)
    company = company.strip()
    
    # Validate company name quality
    if not company or len(company) < 3:
        return None
    
    # Reject if too long (likely description text)
    if len(company) > 80:
        return None
    
    # Remove if it looks invalid
    invalid_patterns = [
        r'^(not specified|unknown|n/a|na|none|company|employer|not available)$',
        r'^[\*\-\s]+$',  # Only asterisks, dashes, or spaces
        r'\b(transforming|leveraging|integrating|facilitating)\b',  # Contains verbs (likely description)
    ]
    for pattern in invalid_patterns:
        if re.search(pattern, company, re.IGNORECASE):
            return None
    
    return company


def clean_summary_text(text: Optional[str]) -> str:
    """
    Clean summary text to remove markdown formatting inconsistencies like "Name**: Value".
    
    Args:
        text: Raw summary text that may contain markdown formatting
        
    Returns:
        Cleaned summary text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code fences
    text = re.sub(r'^```[\w]*\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "Name**:", "**Name**:", "Name:", etc. at the start of lines
    # This handles cases like "Name**: Clarity" or "**Company**: ABC Corp"
    text = re.sub(r'^(\*{0,2}(?:Name|Company|Title|Job Title|Position|Role|Location|Salary|Description|Summary|Employer|Organization)\*{0,2}:?\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove bold markdown (**text** or __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'__([^_]+)__', r'\1', text)  # __text__ -> text
    
    # Remove italic markdown (*text* or _text_)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)  # *text* -> text (but not **text**)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)  # _text_ -> text (but not __text__)
    
    # Remove standalone asterisks at line starts/ends
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\*+$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "**:**" or "**: " at line starts
    text = re.sub(r'^\*{1,2}:?\s*', '', text, flags=re.MULTILINE)
    
    # Clean up multiple consecutive asterisks
    text = re.sub(r'\*{3,}', '', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown list markers that might be left over
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace (multiple spaces/newlines)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Final strip
    text = text.strip()
    
    return text


def extract_job_title_from_content(content: str, fallback_title: Optional[str] = None) -> Optional[str]:
    """
    Extract job title from scraped content using multiple strategies.
    
    Args:
        content: Scraped job content
        fallback_title: Title to use if extraction fails
        
    Returns:
        Extracted job title or fallback message
    """
    if not content:
        return clean_job_title(fallback_title) or "Job title not available in posting"
    
    content_lower = content.lower()
    
    # Pattern 1: Look for "Job Title:", "Position:", "Role:" patterns
    patterns = [
        r'(?:job\s*title|position|role|title)[:\s]+([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})',
        r'(?:we\s+are\s+hiring|looking\s+for|seeking)\s+(?:a|an)?\s*([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})',
        r'^([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})\s+(?:position|role|job|opening)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            potential_title = match.group(1).strip()
            cleaned = clean_job_title(potential_title)
            if cleaned and len(cleaned) >= 5:
                return cleaned
    
    # Pattern 2: Look for common job title keywords followed by text
    job_keywords = [
        r'(senior|junior|lead|principal)?\s*(software|web|frontend|backend|full.?stack|mobile|devops|data|ml|ai|machine learning|artificial intelligence)\s+(engineer|developer|architect|scientist|analyst)',
        r'(product|project|program|engineering|technical|software|data|business|marketing|sales|operations|hr|human resources)\s+(manager|director|lead|specialist|coordinator|assistant|officer|executive)',
        r'(senior|junior|lead|principal)?\s*(designer|developer|engineer|analyst|scientist|consultant|advisor|specialist)',
    ]
    
    for pattern in job_keywords:
        matches = re.finditer(pattern, content[:2000], re.IGNORECASE)
        for match in matches:
            potential_title = match.group(0).strip()
            cleaned = clean_job_title(potential_title)
            if cleaned and len(cleaned) >= 5:
                return cleaned
    
    # Pattern 3: Try to extract from first line or heading
    lines = content.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if len(line) >= 10 and len(line) <= 100:
            # Check if it looks like a job title
            if any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'director', 'executive', 'coordinator', 'officer', 'assistant']):
                cleaned = clean_job_title(line)
                if cleaned:
                    return cleaned
    
    # Fallback to provided title if available
    if fallback_title:
        cleaned = clean_job_title(fallback_title)
        if cleaned:
            return cleaned
    
    return "Job title not available in posting"


def is_valid_company_name(name: str) -> bool:
    """
    Validate if extracted text looks like a real company name.
    
    Args:
        name: Potential company name
    
    Returns:
        True if it looks like a valid company name
    """
    if not name or len(name) < 3:
        return False
    
    # Reject if it's too long (likely description text)
    if len(name) > 80:
        return False
    
    # Reject invalid company name words (expanded list)
    invalid_company_words = [
        'hirer', 'employer', 'recruiter', 'hiring', 'company', 'organization', 'organisation',
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'by', 
        'leveraging', 'transforming', 'using', 'through', 'description', 'about',
        'job', 'position', 'role', 'opportunity', 'career', 'work', 'employment',
        'posting', 'listing', 'advertisement', 'ad', 'vacancy', 'opening',
        'applicant', 'candidate', 'worker', 'employee', 'staff', 'personnel',
        'skip to main content', 'skip navigation', 'skip to content', 'main content',
        'navigation', 'menu', 'home', 'about us', 'contact us', 'privacy policy',
        'terms of service', 'cookie policy', 'accessibility', 'sitemap'
    ]
    name_lower = name.lower().strip()
    # Reject if the entire name is just an invalid word
    if name_lower in invalid_company_words:
        return False
    
    # Reject if it contains invalid words (even as part of the name) - strict check for common false positives
    for invalid_word in ['hirer', 'employer', 'recruiter', 'hiring']:
        # Match whole words only, and reject if it's a short name containing these
        if re.search(r'\b' + re.escape(invalid_word) + r'\b', name_lower):
            # Reject if it's a short name (likely false positive)
            if len(name_lower.split()) <= 2:
                return False
    
    # Reject if it contains too many common words (likely description)
    common_words = ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'by', 'leveraging', 'transforming', 'using', 'through']
    word_count = sum(1 for word in common_words if word in name_lower)
    if word_count >= 3:
        return False
    
    # Reject if it starts with lowercase (likely mid-sentence)
    if name[0].islower():
        return False
    
    # Reject if it contains verbs indicating it's a description
    verb_patterns = [
        r'\b(transforming|leveraging|integrating|facilitating|building|creating|developing|providing)\b',
        r'\b(we|our|their|its)\b',
    ]
    for pattern in verb_patterns:
        if re.search(pattern, name.lower()):
            return False
    
    # Require at least one capital letter (proper noun indicator)
    if not any(c.isupper() for c in name):
        return False
    
    # Reject if it's a sentence fragment (contains sentence-ending punctuation)
    if re.search(r'[.!?]\s*$', name):
        return False
    
    return True


def extract_company_name_from_content(content: str, fallback_company: Optional[str] = None) -> Optional[str]:
    """
    [LEGACY] Extract company name from scraped content using regex patterns.
    
    NOTE: This function is kept for backward compatibility and fallback scenarios.
    Primary extraction now uses Gemini API via extract_company_and_title_from_raw_data().
    
    Args:
        content: Scraped job content
        fallback_company: Company name to use if extraction fails
        
    Returns:
        Extracted company name or fallback message
    """
    if not content:
        return clean_company_name(fallback_company) or "Company name not available in posting"
    
    content_lower = content.lower()
    
    # Pattern 1: Look for explicit company labels with proper names
    # Prioritize patterns with company suffixes (Ltd, Inc, etc.)
    priority_patterns = [
        r'(?:company|employer|organization|organisation)(?:\s+description)?[:\s]+([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Ltd|Limited|Inc|LLC|Corp|Corporation|Group|Holdings|Technology|Solutions|Services|Pvt\.?\s*Ltd\.?))',
        r'([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Pvt\.?\s*Ltd\.?|Private Limited|Ltd\.?|Limited|Inc\.?|LLC|Corporation|Corp\.?))',
        r'(?:at|for|with)\s+([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Ltd|Limited|Inc|LLC|Corp|Corporation|Group|Holdings|Technology|Solutions|Services|Pvt\.?\s*Ltd\.?))',
    ]
    
    for pattern in priority_patterns:
        matches = re.finditer(pattern, content[:1500])
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Pattern 2: Look for "Company Description" or "About" sections
    section_patterns = [
        r'(?:company\s+description|about\s+(?:the\s+)?company|about\s+us)[:\s]+([A-Z][A-Za-z0-9\s&.,\-\']{3,60}?)(?:\s+is|\s+integrate|\s+provide|\.|,)',
        r'(?:at|join|work\s+at|careers\s+at)\s+([A-Z][A-Za-z0-9\s&.,\-\']{3,50}?)(?:\s*,|\s+is|\s+we)',
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, content[:1000], re.IGNORECASE)
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Pattern 3: Look for "by [Company]" pattern (common in job listings)
    by_patterns = [
        r'(?:by|from)\s+([A-Z][A-Za-z0-9\s&.,\-\']{3,50}?)(?:\s+is|\s+integrate|\s+provide|\s*\n|\s*$)',
    ]
    
    for pattern in by_patterns:
        matches = re.finditer(pattern, content[:500])
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Fallback to provided company if available
    if fallback_company:
        cleaned = clean_company_name(fallback_company)
        if cleaned and is_valid_company_name(cleaned):
            return cleaned
    
    return "Company name not available in posting"



def extract_json_from_response(text: str) -> Dict[str, Any]:

    """Extract JSON from agent response, handling markdown code blocks and nested content."""

    if not text:

        return {}

    

    original_text = text

    text = text.strip()

    

    # Handle phi agent response objects and other response types

    if hasattr(text, 'content'):

        text = str(text.content)

    elif hasattr(text, 'messages') and text.messages:

        # Get last message content

        last_msg = text.messages[-1]

        text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)

    else:

        text = str(text)

    

    text = text.strip()

    

    # Remove markdown code fences - more comprehensive matching

    if '```json' in text:

        # Extract content between ```json and ```

        match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)

        if match:

            text = match.group(1).strip()

    elif '```' in text:

        # Remove any code fence markers

        lines = text.split("\n")

        start_idx = 0

        end_idx = len(lines)

        

        # Find first non-code-fence line

        for i, line in enumerate(lines):

            if line.strip().startswith("```"):

                start_idx = i + 1

                break

        

        # Find last code-fence line

        for i in range(len(lines) - 1, -1, -1):

            if lines[i].strip() == "```" or lines[i].strip().startswith("```"):

                end_idx = i

                break

        

        text = "\n".join(lines[start_idx:end_idx]).strip()

    

    # Clean up common artifacts

    text = re.sub(r'^[^{]*', '', text)  # Remove leading non-JSON text

    text = re.sub(r'[^}]*$', '', text)  # Remove trailing non-JSON text

    text = text.strip()

    

    # Try direct JSON parse first

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError as e:

        pass

    

    # Try to fix common JSON issues and parse again

    fixed_text = text

    

    # Fix trailing commas before closing braces/brackets

    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)

    

    # Try parsing after trailing comma fix

    try:

        parsed = json.loads(fixed_text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass

    

    # Try to find the largest valid JSON object in the text

    # Find all potential JSON object boundaries

    start_positions = [m.start() for m in re.finditer(r'\{', text)]

    end_positions = [m.start() for m in re.finditer(r'\}', text)]

    

    # Try parsing from each opening brace

    best_match = None

    best_length = 0

    

    for start_pos in start_positions:

        # Find matching closing brace

        brace_count = 0

        for i in range(start_pos, len(text)):

            if text[i] == '{':

                brace_count += 1

            elif text[i] == '}':

                brace_count -= 1

                if brace_count == 0:

                    # Found matching brace

                    candidate = text[start_pos:i+1]

                    try:

                        parsed = json.loads(candidate)

                        if isinstance(parsed, dict) and len(parsed) > best_length:

                            best_match = parsed

                            best_length = len(parsed)

                    except json.JSONDecodeError:

                        pass

                    break

    

    if best_match:

        return best_match

    

    # Last resort: try to extract key-value pairs using regex

    result = {}

    # Extract quoted keys and values

    kv_pattern = r'"([^"]+)":\s*([^,}\]]+)'

    matches = re.finditer(kv_pattern, text)

    for match in matches:

        key = match.group(1)

        value = match.group(2).strip()

        # Try to parse value

        if value.startswith('"') and value.endswith('"'):

            result[key] = value[1:-1]

        elif value.startswith('['):

            # Try to parse array

            try:

                result[key] = json.loads(value)

            except:

                result[key] = value

        elif value.lower() in ('true', 'false'):

            result[key] = value.lower() == 'true'

        elif value.isdigit():

            result[key] = int(value)

        elif re.match(r'^\d+\.\d+$', value):

            result[key] = float(value)

        else:

            result[key] = value

    

    if result:

        print(f"⚠️  Partially parsed JSON using regex fallback. Got {len(result)} fields.")

        return result

    

    # If all else fails, log and return empty dict (workflow should handle this)

    print(f"⚠️  Failed to parse JSON from response")

    print(f"Response length: {len(original_text)} chars")

    print(f"Response preview: {original_text[:500]}...")

    return {}





def parse_experience_years(value: Any) -> Optional[float]:

    """Parse total years of experience from various formats."""

    if value is None:

        return None

    

    if isinstance(value, (int, float)):

        return float(value)

    

    if isinstance(value, str):

        # Extract numbers from strings like "1 year", "2-3 years", "1.5 years"

        numbers = re.findall(r'\d+\.?\d*', value)

        if numbers:

            try:

                return float(numbers[0])

            except:

                pass

    

    return None





def detect_portal(url: str) -> str:

    """Detect the job portal from URL domain."""

    url_lower = url.lower()

    if 'linkedin.com' in url_lower:

        return 'LinkedIn'

    elif 'internshala.com' in url_lower:

        return 'Internshala'

    elif 'indeed.com' in url_lower:

        return 'Indeed'

    elif 'glassdoor.com' in url_lower:

        return 'Glassdoor'

    elif 'monster.com' in url_lower:

        return 'Monster'

    elif 'naukri.com' in url_lower:

        return 'Naukri'

    elif 'timesjobs.com' in url_lower:

        return 'TimesJobs'

    elif 'shine.com' in url_lower:

        return 'Shine'

    elif 'hired.com' in url_lower:

        return 'Hired'

    elif 'angel.co' in url_lower or 'angelist.com' in url_lower:

        return 'AngelList'

    elif 'stackoverflow.com' in url_lower or 'stackoverflowjobs.com' in url_lower:

        return 'Stack Overflow'

    elif 'github.com' in url_lower:

        return 'GitHub Jobs'

    elif 'dice.com' in url_lower:

        return 'Dice'

    elif 'ziprecruiter.com' in url_lower:

        return 'ZipRecruiter'

    elif 'simplyhired.com' in url_lower:

        return 'SimplyHired'

    else:

        # Extract domain name as fallback

        try:

            from urllib.parse import urlparse

            parsed = urlparse(url)

            domain = parsed.netloc.replace('www.', '').split('.')[0]

            return domain.capitalize()

        except:

            return 'Unknown'





def extract_json_ld_job_title(soup: BeautifulSoup) -> Optional[str]:

    """Extract job title from JSON-LD structured data."""

    try:

        for script in soup.find_all('script', type=lambda t: t and 'json' in str(t).lower() and 'ld' in str(t).lower()):

            try:

                json_data = json.loads(script.string or '{}')

                

                def extract_from_obj(obj):

                    if isinstance(obj, dict):

                        obj_type = obj.get('@type', '')

                        if 'JobPosting' in str(obj_type):

                            # Try different field names

                            for field in ['title', 'jobTitle', 'name', 'jobTitleText']:

                                if field in obj and obj[field]:

                                    return str(obj[field]).strip()

                        # Recursively search nested objects

                        for value in obj.values():

                            result = extract_from_obj(value)

                            if result:

                                return result

                    elif isinstance(obj, list):

                        for item in obj:

                            result = extract_from_obj(item)

                            if result:

                                return result

                    return None

                

                result = extract_from_obj(json_data)

                if result:

                    return result

            except (json.JSONDecodeError, AttributeError):

                continue

    except Exception:

        pass

    return None





def extract_job_info_from_url(url: str, firecrawl_api_key: Optional[str] = None) -> Dict[str, Any]:

    """

    Extract job title, company name from a job URL.

    Reuses the scraping logic from fetch_job function with enhanced extraction.

    

    Returns:

        Dictionary with 'job_title', 'company_name', 'portal', and 'success' fields

    """

    try:

        # Detect portal first

        portal = detect_portal(url)

        

        # Use Firecrawl SDK directly

        fc = scrape_website_custom(url, firecrawl_api_key)

        content = ''

        title = ''

        company = ''

        html_content = ''

        

        if isinstance(fc, dict) and 'error' not in fc:

            content = str(fc.get('content') or fc.get('markdown') or fc)

            md = fc.get('metadata') or {}

            title = md.get('title') or ''

            html_content = fc.get('html') or ''



        # Always parse HTML for better title/company extraction

        if not requests or not BeautifulSoup:

            return {

                'job_url': url,

                'job_title': None,

                'company_name': None,

                'portal': portal,

                'visa_scholarship_info': "Not specified",
                'success': False,

                'error': 'requests and beautifulsoup4 are required for HTML parsing'

            }

        

        if not html_content:

            headers = {

                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',

                'Accept-Language': 'en-US,en;q=0.9',

            }

            resp = requests.get(url, headers=headers, timeout=20)

            if resp.ok:

                html_content = resp.text

                soup = BeautifulSoup(html_content, 'lxml')

            else:

                return {

                    'job_url': url,

                    'job_title': None,

                    'company_name': None,

                    'portal': portal,

                    'visa_scholarship_info': "Not specified",
                    'success': False,

                    'error': f'Failed to fetch URL: {resp.status_code}'

                }

        else:

            soup = BeautifulSoup(html_content, 'lxml')

        

        # Enhanced title extraction - try multiple methods in order of accuracy

        

        # 1. JSON-LD structured data (most reliable)

        if not title:

            title = extract_json_ld_job_title(soup)

        

        # 2. Portal-specific selectors

        if not title:

            portal_lower = portal.lower()

            if portal_lower == 'internshala':

                # Internshala specific selectors

                title_elem = soup.select_one('.profile, .job_title, h1.profile_on_detail_page, .heading_4_5')

                if title_elem:

                    title = title_elem.get_text(strip=True)

            elif portal_lower == 'linkedin':

                # LinkedIn specific selectors

                title_elem = soup.select_one('.jobs-details-top-card__job-title, h1[data-test-id*="job-title"], .topcard__title')

                if title_elem:

                    title = title_elem.get_text(strip=True)

            elif portal_lower == 'indeed':

                # Indeed specific selectors

                title_elem = soup.select_one('.jobsearch-JobInfoHeader-title, h2.jobTitle')

                if title_elem:

                    title = title_elem.get_text(strip=True)

        

        # 3. Common job title selectors (expanded list)

        if not title:

            job_title_selectors = [

                # Class-based selectors

                'h1.job-title', 'h2.job-title', '.job-title', '.jobTitle', '.jobtitle',

                '[class*="job-title"]', '[class*="JobTitle"]', '[class*="jobTitle"]',

                '[data-testid*="job-title"]', '[data-testid*="jobTitle"]',

                '[data-cy*="job-title"]', '[data-job-title]',

                # ID-based selectors

                '#job-title', '#jobTitle', '#job_title',

                # Semantic selectors

                'h1[itemprop="title"]', '[itemprop="jobTitle"]',

                'h1[role="heading"]', '.heading-title',

                # Generic headings (check if they look like job titles)

                'h1', 'h2.title', '.title'

            ]

            for selector in job_title_selectors:

                try:

                    elem = soup.select_one(selector)

                    if elem:

                        title_text = elem.get_text(strip=True)

                        # Validate it looks like a job title

                        if title_text and len(title_text) < 150 and len(title_text) > 3:

                            # Exclude common non-job-title patterns

                            if not any(skip in title_text.lower() for skip in ['home', 'about', 'contact', 'login', 'sign up', 'menu', 'navigation']):

                                title = title_text

                                break

                except Exception:

                    continue

        

        # 4. Meta tags

        if not title or len(title) > 150:

            # Open Graph title

            og_title = soup.find('meta', property='og:title')

            if og_title and og_title.get('content'):

                og_title_text = og_title.get('content').strip()

                if og_title_text and len(og_title_text) < 150:

                    title = og_title_text

            

            # Twitter card title

            if not title or len(title) > 150:

                twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})

                if twitter_title and twitter_title.get('content'):

                    twitter_title_text = twitter_title.get('content').strip()

                    if twitter_title_text and len(twitter_title_text) < 150:

                        title = twitter_title_text

            

            # Schema.org itemprop

            if not title:

                itemprop_title = soup.find(attrs={'itemprop': 'title'})

                if itemprop_title:

                    title = itemprop_title.get_text(strip=True)

        

        # 5. Page title as fallback (with better cleaning)

        if not title:

            if soup.title and soup.title.string:

                page_title = soup.title.string.strip()

                # Clean common prefixes/suffixes

                page_title = re.sub(r'^\s*[-|]\s*', '', page_title)  # Remove leading separators

                page_title = re.sub(r'\s*[-|]\s*$', '', page_title)  # Remove trailing separators

                # Remove common website suffixes

                page_title = re.sub(r'\s*-?\s*(LinkedIn|Indeed|Glassdoor|Monster|Internshala).*$', '', page_title, flags=re.I)

                if page_title and len(page_title) < 150:

                    title = page_title

        

        # 6. Extract from first heading if still not found

        if not title:

            h1 = soup.find('h1')

            if h1:

                h1_text = h1.get_text(strip=True)

                if h1_text and len(h1_text) < 150 and len(h1_text) > 3:

                    title = h1_text

        

        # Extract company name - try multiple sources (same logic as fetch_job)

        def has_company_class(class_attr):

            if not class_attr:

                return False

            if isinstance(class_attr, list):

                return any('company' in str(c).lower() for c in class_attr)

            return 'company' in str(class_attr).lower()

        

        if not company:

            company_selectors = [

                '.company-name', '[class*="Company"]', 

                '[data-testid*="company"]', 'a[href*="/company/"]',

                'strong', '.employer'

            ]

            for selector in company_selectors:

                elem = soup.select_one(selector)

                if elem:

                    company_text = elem.get_text(strip=True)

                    if company_text and 3 <= len(company_text) <= 50:

                        company = company_text

                        break

            

            # Try elements with common class names

            if not company:

                for tag in ['span', 'div', 'a', 'p', 'h3', 'h4']:

                    elements = soup.find_all(tag, class_=has_company_class)

                    for elem in elements[:3]:

                        company_text = elem.get_text(strip=True)

                        if company_text and 3 <= len(company_text) <= 50:

                            company = company_text

                            break

                    if company:

                        break

            

            # Try strong tags with company/employer text

            if not company:

                strong_tags = soup.find_all('strong')

                for strong in strong_tags:

                    strong_text = strong.get_text(strip=True).lower()

                    if 'company' in strong_text or 'employer' in strong_text:

                        parent = strong.find_parent()

                        if parent:

                            parent_text = parent.get_text(strip=True)

                            if len(parent_text) < 100:

                                company = parent_text

                                break

            

            # Try meta tags

            if not company:

                meta_tags = soup.find_all('meta')

                for meta in meta_tags:

                    name_attr = meta.get('name', '').lower()

                    if name_attr and ('company' in name_attr or 'employer' in name_attr):

                        content_meta = meta.get('content', '').strip()
                        if content_meta and 3 <= len(content_meta) <= 50:
                            company = content_meta
                            break

            

            # Look in content text for "at [Company]" pattern

            if not company and content:

                company_match = re.search(r'\bat\s+([A-Z][A-Za-z\s&]{2,40})\b', content[:1000], re.I)

                if company_match:

                    company = company_match.group(1).strip()

        

        # Enhanced title cleaning and validation

        if title:

            # Remove common suffixes/prefixes

            title = re.sub(r'\s*[-–—|]\s*at\s+[^-]+$', '', title, flags=re.I)  # Remove " - at Company Name"

            title = re.sub(r'\s*[-–—|]\s*[^-]+(?:\.com|\.in|\.org).*$', '', title, flags=re.I)  # Remove website suffixes

            title = re.sub(r'\s*[-–—|]\s*.+$', '', title)  # Remove " - Company Name" (generic)

            title = re.sub(r'^[^:]*:\s*', '', title)  # Remove "Job Board: "

            title = re.sub(r'\s*[|]\s*', ' ', title)  # Replace pipe separators with space

            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace

            title = title.strip()

            

            # Validate title quality

            if title:

                # Remove if it's too short or looks like navigation

                if len(title) < 3 or len(title) > 150:

                    title = None

                elif any(bad in title.lower() for bad in ['home', 'menu', 'navigation', 'skip to', 'cookie', 'privacy policy']):

                    title = None

            

            if title:

                title = title[:100]  # Limit length

        

        if company:

            company = company.strip()[:50]  # Limit length

            company = re.sub(r'^at\s+', '', company, flags=re.I)

            company = company.strip()

        

        visa_scholarship_info: Optional[str] = None
        visa_keywords = [
            "visa sponsorship",
            "visa support",
            "scholarship",
            "h1b",
            "work permit",
            "financial support",
            "tuition assistance",
            "visa assistance",
        ]
        search_sources: List[str] = []
        if content:
            search_sources.append(str(content))
        try:
            search_sources.append(soup.get_text(separator=' ', strip=True))
        except Exception:
            pass
        for raw_text in search_sources:
            lower_text = raw_text.lower()
            if any(keyword in lower_text for keyword in visa_keywords):
                for keyword in visa_keywords:
                    if keyword in lower_text:
                        idx = lower_text.find(keyword)
                        start = max(0, idx - 100)
                        end = min(len(raw_text), idx + len(keyword) + 200)
                        visa_scholarship_info = raw_text[start:end].strip()
                        break
            if visa_scholarship_info:
                break
        if not visa_scholarship_info:
            visa_scholarship_info = "Not specified"
        
        return {

            'job_url': url,

            'job_title': title or None,

            'company_name': company or None,

            'portal': portal,

            'visa_scholarship_info': visa_scholarship_info,
            'success': True,

            'error': None

        }

        

    except Exception as e:

        return {

            'job_url': url,

            'job_title': None,

            'company_name': None,

            'portal': detect_portal(url),

            'visa_scholarship_info': "Not specified",
            'success': False,

            'error': str(e)

        }





@app.get("/api/progress/{request_id}", response_model=ProgressStatus)

async def get_progress(request_id: str):

    status = REQUEST_PROGRESS.get(request_id)

    if not status:

        raise HTTPException(status_code=404, detail="Unknown request_id")

    return status





# Background task functions for Firebase saves
def save_job_applications_background(user_id: str, jobs_to_save: List[Dict[str, Any]]):
    """Background task to save job applications to Firebase (non-blocking)."""
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        saved_doc_ids = firebase_service.save_job_applications_batch(user_id, jobs_to_save)
        logger.info(f"Background save completed: {len(saved_doc_ids)} job applications saved for user {user_id}")
    except ImportError as e:
        logger.warning(f"Firebase service not available for background save: {e}")
    except Exception as e:
        logger.error(f"Background save failed for job applications: {e}", exc_info=True)


def save_sponsorship_info_background(
    user_id: str,
    request_id: str,
    sponsorship_data: Dict[str, Any],
    job_info: Optional[Dict[str, Any]] = None
):
    """Background task to save sponsorship info to Firebase (non-blocking)."""
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        doc_id = firebase_service.save_sponsorship_info(
            user_id=user_id,
            request_id=request_id,
            sponsorship_data=sponsorship_data,
            job_info=job_info
        )
        logger.info(f"Background save completed: sponsorship info saved with doc_id {doc_id} for user {user_id}")
    except ImportError as e:
        logger.warning(f"Firebase service not available for background save: {e}")
    except Exception as e:
        logger.error(f"Background save failed for sponsorship info: {e}", exc_info=True)


@app.post("/api/match-jobs", response_model=MatchJobsResponse, dependencies=[Depends(rate_limit)])

async def match_jobs(

    json_body: Optional[str] = Form(default=None),

    pdf_file: Optional[UploadFile] = File(default=None),

    settings: Settings = Depends(get_settings),

    background_tasks: BackgroundTasks = BackgroundTasks(),

):

    request_id = make_request_id()
    start_time = time.time()  # Track processing time

    REQUEST_PROGRESS[request_id] = ProgressStatus(

        request_id=request_id, status="queued", jobs_total=0, jobs_scraped=0, 

        jobs_cached=0, started_at=now_iso(), updated_at=now_iso()

    )



    try:

        # Parse input - support new format with jobs field, legacy format, and old format

        data: Optional[MatchJobsRequest] = None

        legacy_data: Optional[MatchJobsJsonRequest] = None

        new_format_jobs: Optional[Dict[str, Any]] = None  # New format with jobtitle, joblink, jobdata
        jobs_string: Optional[str] = None  # New format with jobs as string (HTML/text content)

        user_id: Optional[str] = None

        

        if json_body:

            try:

                # Handle JSON that might be double-encoded or have extra quotes

                clean_json = json_body.strip()

                if clean_json.startswith('"') and clean_json.endswith('"'):

                    clean_json = clean_json[1:-1].replace('\\"', '"')

                payload = json.loads(clean_json)

                
                
                # Debug: Log what we received
                print(f"\n[REQUEST FORMAT DETECTION]")
                print(f"Payload keys: {list(payload.keys())}")
                if "jobs" in payload:
                    print(f"jobs type: {type(payload['jobs'])}")
                    if isinstance(payload["jobs"], dict):
                        print(f"jobs dict keys: {list(payload['jobs'].keys())}")
                    elif isinstance(payload["jobs"], list):
                        print(f"jobs list length: {len(payload['jobs'])}")
                    elif isinstance(payload["jobs"], str):
                        print(f"jobs string length: {len(payload['jobs'])} characters")
                

                # Check for new format with jobs as string (HTML/text content)
                if "jobs" in payload and isinstance(payload["jobs"], str):
                    jobs_string = payload["jobs"]
                    user_id = payload.get("user_id")
                    print(f"[STRING FORMAT] Detected jobs as string (HTML/text content), length: {len(jobs_string)}")

                # Check for new format with jobs field (jobtitle, joblink, jobdata)
                elif "jobs" in payload and isinstance(payload["jobs"], dict):

                    new_format_jobs = payload["jobs"]

                    user_id = payload.get("user_id")

                    print(f"[NEW FORMAT] Detected jobs field with jobtitle, joblink, jobdata")

                # Try new format first (resume + jobs list)

                elif "resume" in payload and "jobs" in payload:

                    print(f"[STANDARD FORMAT] Detected resume + jobs list format")
                    data = MatchJobsRequest(**payload)

                else:

                    # Legacy format

                    print(f"[LEGACY FORMAT] Attempting to parse as legacy format with urls")
                    try:
                        legacy_data = MatchJobsJsonRequest(**payload)
                        print(f"[LEGACY FORMAT] Successfully parsed. URLs count: {len(legacy_data.urls) if legacy_data and hasattr(legacy_data, 'urls') else 0}")
                    except Exception as legacy_error:
                        print(f"[LEGACY FORMAT] Failed to parse as legacy format: {legacy_error}")
                        # Check if it's a validation error about missing URLs
                        if "at least one" in str(legacy_error).lower() or "required" in str(legacy_error).lower():
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid request format: No job URLs found. Expected one of: (1) 'jobs' dict with 'jobtitle', 'joblink', 'jobdata' (new format), (2) 'resume' + 'jobs' list (standard format), or (3) 'urls' list with at least one URL (legacy format). Error: {legacy_error}"
                            )
                        raise

            except HTTPException:
                # Re-raise HTTPExceptions as-is
                raise
            except Exception as e:

                raise HTTPException(

                    status_code=400, 

                    detail=f"Invalid JSON body: {e}. Received: {json_body[:200] if json_body else 'None'}"

                )

        else:

            raise HTTPException(status_code=400, detail="Missing json_body field")



        REQUEST_PROGRESS[request_id].status = "parsing"

        REQUEST_PROGRESS[request_id].updated_at = now_iso()



        # Get resume text

        resume_bytes: Optional[bytes] = None

        if data and data.resume and data.resume.content:

            resume_bytes = decode_base64_pdf(data.resume.content)

        elif legacy_data and legacy_data.pdf:

            resume_bytes = decode_base64_pdf(legacy_data.pdf)

        elif pdf_file is not None:

            try:

                pdf_file.file.seek(0)

            except Exception:

                pass

            resume_bytes = await pdf_file.read()



        if not resume_bytes:

            raise HTTPException(

                status_code=400, 

                detail="Missing resume PDF (base64 or file upload)"

            )



        resume_text = extract_text_from_pdf_bytes(resume_bytes)



        # Set environment variables for agents

        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")

        os.environ.setdefault("FIRECRAWL_API_KEY", settings.firecrawl_api_key or "")



        # STEP 1: Parse Resume (OCR extracts text, then LLM parses it)
        logger.info("Step 1: OCR extracted text from PDF, now parsing with LLM")
        
        from resume_parser_ocr import parse_resume_with_llm_fallback
        
        # Extract complete text from PDF using OCR (already done above with extract_text_from_pdf_bytes)
        # Now give that complete OCR-extracted text directly to LLM for parsing
        resume_json = await asyncio.to_thread(
            parse_resume_with_llm_fallback,
            resume_text,  # Complete OCR-extracted text from PDF
            settings.model_name,
            settings.openai_api_key
        )
        
        try:
            # Validate we got something useful

            

            # Validate we got something useful
            if not resume_json or not resume_json.get("name") or resume_json.get("name") == "Unknown":
                logger.warning("Resume parsing returned incomplete data, using minimal fallback")
                resume_json = {
                    "name": "Unknown Candidate",
                    "email": None,
                    "phone": None,
                    "skills": [],
                    "experience_summary": resume_text[:500],
                    "total_years_experience": 1.0,
                    "education": [],
                    "certifications": [],
                    "interests": []
                }
        except Exception as e:
            logger.error(f"Error parsing resume: {e}", exc_info=True)
            # Last resort fallback
            resume_json = {
                "name": "Unknown Candidate",
                "email": None,
                "phone": None,
                "skills": [],
                "experience_summary": resume_text[:500],
                "total_years_experience": None,
                "education": [],
                "certifications": [],
                "interests": []
            }



        # Handle experience_summary - convert to string if needed

        exp_summary = resume_json.get("experience_summary")

        if isinstance(exp_summary, (list, dict)):

            exp_summary = json.dumps(exp_summary, indent=2)

        elif exp_summary is None:

            exp_summary = "Not provided"

        

        # Parse total years of experience

        total_years = parse_experience_years(resume_json.get("total_years_experience"))

        

        candidate_profile = CandidateProfile(

            name=resume_json.get("name") or "Unknown",

            email=resume_json.get("email"),

            phone=resume_json.get("phone"),

            skills=resume_json.get("skills", []) or [],

            experience_summary=exp_summary,

            total_years_experience=total_years,

            interests=resume_json.get("interests", []) or [],

            education=resume_json.get("education", []) or [],

            certifications=resume_json.get("certifications", []) or [],

            raw_text_excerpt=redact_long_text(resume_text, 300),

        )



        # STEP 2: Prepare job URLs or use new format with jobdata

        jobs: List[JobPosting] = []

        urls: List[str] = []  # Initialize urls to avoid UnboundLocalError

        
        # Debug: Log which format was detected
        print(f"\n[FORMAT DETECTION RESULT]")
        print(f"jobs_string: {jobs_string is not None}")
        print(f"new_format_jobs: {new_format_jobs is not None}")
        print(f"data: {data is not None}")
        print(f"legacy_data: {legacy_data is not None}")
        if legacy_data:
            print(f"legacy_data.urls: {legacy_data.urls if hasattr(legacy_data, 'urls') else 'N/A'}")
        if data:
            print(f"data.jobs: {len(data.jobs) if hasattr(data, 'jobs') and data.jobs else 0} jobs")

        if jobs_string:
            # STRING FORMAT: Process jobs as HTML/text string with summarizer
            print("\n" + "="*80)
            print(f"📝 SUMMARIZER - Processing job data from string (string format)")
            print("="*80)
            
            from scrapers.response import summarize_scraped_data
            
            # Create scraped_data structure for summarizer from the string
            scraped_data = {
                "url": "https://example.com",  # No URL available for string format
                "job_title": None,
                "company_name": None,
                "location": None,
                "description": jobs_string,
                "qualifications": None,
                "suggested_skills": None,
                "text_content": jobs_string,
                "html_length": len(jobs_string)
            }
            
            # Use summarizer to process the job data
            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OpenAI API key is required for summarization")
            
            print(f"Processing job from string, length: {len(jobs_string)} characters")
            
            # Run summarizer in thread pool
            summarized_data = await asyncio.to_thread(
                summarize_scraped_data,
                scraped_data,
                openai_key
            )
            
            # Extract job title and company from summarized data
            final_job_title = clean_job_title(summarized_data.get("job_title")) or "Job title not available"
            final_company = clean_company_name(summarized_data.get("company_name")) or "Company name not available"
            
            # Convert experience_level to string if it's a dict
            experience_level = summarized_data.get("required_experience")
            if isinstance(experience_level, dict):
                parts = []
                if "years" in experience_level:
                    parts.append(f"{experience_level['years']} years")
                if "type" in experience_level:
                    parts.append(experience_level["type"])
                experience_level = ", ".join(parts) if parts else str(experience_level)
            elif experience_level is not None:
                experience_level = str(experience_level)
            
            print(f"[Job Info] Final job title: {final_job_title}")
            print(f"[Job Info] Final company: {final_company}")
            
            # Create JobPosting from summarized data
            job = JobPosting(
                url="https://example.com",  # No URL for string format
                job_title=final_job_title,
                company=final_company,
                description=summarized_data.get("description") or jobs_string,
                skills_needed=summarized_data.get("required_skills", []) or [],
                experience_level=experience_level,
                salary=summarized_data.get("salary")
            )
            
            jobs = [job]
            urls = []  # No URLs for string format
            REQUEST_PROGRESS[request_id].jobs_total = 1
            REQUEST_PROGRESS[request_id].jobs_scraped = 1
            REQUEST_PROGRESS[request_id].updated_at = now_iso()
            
        elif new_format_jobs:

            # NEW FORMAT: Use jobdata directly with summarizer (skip scraping)

            print("\n" + "="*80)

            print(f"📝 SUMMARIZER - Processing job data directly (new format)")

            print("="*80)

            

            from scrapers.response import summarize_scraped_data

            

            # Extract job information from new format

            raw_job_title = new_format_jobs.get("jobtitle", "")
            job_link = new_format_jobs.get("joblink", "")

            job_data = new_format_jobs.get("jobdata", "")

            

            # IMPORTANT: Send raw scraped data directly to OpenAI to extract company name and job title
            # Do NOT perform any preprocessing, parsing, or extraction - send it as-is to OpenAI
            logger.debug(f"[EXTRACT] Extracting from {len(job_data)} chars via OpenAI")
            
            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not openai_key:
                logger.warning("OpenAI API key not available, using raw values as fallback")
                extracted_company = None
                extracted_title = None
            else:
                # Use the new function that sends raw data directly to OpenAI
                from job_extractor import extract_company_and_title_from_raw_data
                
                # Extract company name and job title directly from raw scraped data using OpenAI
                # Handle OpenAI errors gracefully - fallback to raw values if extraction fails
                try:
                    extracted_info = await asyncio.to_thread(
                        extract_company_and_title_from_raw_data,
                        job_data,  # Send raw scraped data as-is, no preprocessing
                        openai_key,
                        "gpt-4o-mini"  # Use fast and intelligent OpenAI model
                    )
                    
                    # Get extracted values from OpenAI
                    extracted_company = extracted_info.get("company_name")
                    extracted_title = extracted_info.get("job_title")
                    
                    logger.debug(f"[EXTRACT] OpenAI → Company: {extracted_company[:50] if extracted_company else 'None'}, Title: {extracted_title[:50] if extracted_title else 'None'}")
                except Exception as e:
                    logger.warning(f"OpenAI extraction failed: {e}, using raw values as fallback")
                    extracted_company = None
                    extracted_title = None
            
            # Use extracted values (fallback to raw_job_title if Gemini didn't extract title)
            final_extracted_company = extracted_company
            final_extracted_title = extracted_title or raw_job_title
            
            # Create scraped_data structure for summarizer (using Gemini-extracted values)

            scraped_data = {

                "url": job_link,

                "job_title": final_extracted_title,  # Use Gemini-extracted title
                "company_name": final_extracted_company,  # Use Gemini-extracted company name
                "location": None,

                "description": job_data,  # Keep raw job_data for description

                "qualifications": None,

                "suggested_skills": None,

                "text_content": job_data,  # Keep raw job_data for text_content

                "html_length": len(job_data)

            }

            

            # Skip redundant summarization - we already have company/title extracted
            # The job description will be used as-is for scoring, and we'll summarize later if needed
            summarized_data = {
                "description": job_data,
                "required_skills": [],
                "required_experience": None
            }

            

            # Prioritize raw_job_title if it's provided and looks valid (frontend already has correct title)
            # Only use OpenAI-extracted title if raw_job_title is missing or invalid
            if raw_job_title and len(raw_job_title.strip()) >= 5:
                # Use raw_job_title as primary source, but clean it gently (don't truncate)
                cleaned_raw = clean_job_title(raw_job_title)
                if cleaned_raw and len(cleaned_raw) >= 5:
                    final_job_title = cleaned_raw
                else:
                    # If cleaning broke it, use raw as-is (it's from frontend, likely correct)
                    final_job_title = raw_job_title.strip()
            else:
                # Fallback to OpenAI-extracted title if raw_job_title is not available
                final_job_title = final_extracted_title or summarized_data.get("job_title") or "Job title not available in posting"
                # Clean the extracted title
                if final_job_title and final_job_title != "Job title not available in posting":
                    cleaned_title = clean_job_title(final_job_title)
                    if cleaned_title and len(cleaned_title) >= 5:
                        final_job_title = cleaned_title
            
            final_company = final_extracted_company or summarized_data.get("company_name") or "Company name not available in posting"
            
            if final_company and final_company != "Company name not available in posting":
                cleaned_company = clean_company_name(final_company)
                if cleaned_company and len(cleaned_company) >= 2 and cleaned_company.lower() not in ["not specified", "unknown", "none"]:
                    final_company = cleaned_company
                else:
                    final_company = "Company name not available in posting"
            
            print(f"[FINAL] Title: {final_job_title[:60]}, Company: {final_company[:60]}")
            
            # Convert experience_level to string if it's a dict
            experience_level = summarized_data.get("required_experience")
            if isinstance(experience_level, dict):
                # Convert dict to readable string
                parts = []
                if "years" in experience_level:
                    parts.append(f"{experience_level['years']} years")
                if "type" in experience_level:
                    parts.append(experience_level["type"])
                experience_level = ", ".join(parts) if parts else str(experience_level)
            elif experience_level is not None:
                experience_level = str(experience_level)
            
            # Create JobPosting from summarized data

            job = JobPosting(

                url=job_link if job_link else "https://example.com",

                job_title=final_job_title,
                company=final_company,
                description=summarized_data.get("description") or job_data,

                skills_needed=summarized_data.get("required_skills", []) or [],

                experience_level=experience_level,
                salary=summarized_data.get("salary")

            )

            

            jobs = [job]

            urls = [job_link] if job_link else []  # Set urls for response tracking

            REQUEST_PROGRESS[request_id].jobs_total = 1

            REQUEST_PROGRESS[request_id].jobs_scraped = 1

            REQUEST_PROGRESS[request_id].updated_at = now_iso()

            

        else:

            # OLD FORMAT: Scrape jobs as before

            if data:

                urls = [str(job.url) for job in data.jobs]

                job_titles = {str(job.url): job.title for job in data.jobs}

                job_companies = {str(job.url): job.company for job in data.jobs}

            elif legacy_data:

                urls = [str(u) for u in legacy_data.urls]

                job_titles = {}

                job_companies = {}

            else:

                # Neither data nor legacy_data was set - this shouldn't happen, but handle it gracefully

                raise HTTPException(

                    status_code=400,

                    detail="Invalid request format: No jobs or URLs found in request. Expected 'jobs' dict (new format), 'resume' + 'jobs' list (standard format), or 'urls' list (legacy format)."

                )

            

            # Validate that we have URLs to scrape

            if not urls or len(urls) == 0:

                raise HTTPException(

                    status_code=400,

                    detail=f"No job URLs provided. Received {len(urls)} URLs. Please include job URLs in your request."

                )

                

            REQUEST_PROGRESS[request_id].status = "scraping"

            REQUEST_PROGRESS[request_id].jobs_total = len(urls)

            REQUEST_PROGRESS[request_id].updated_at = now_iso()



            # STEP 3: Scrape Jobs

            print("\n" + "="*80)

            print(f"🔍 JOB SCRAPER - Fetching {len(urls)} job postings")

            print("="*80)



            semaphore = asyncio.Semaphore(settings.max_concurrent_scrapes)



            async def fetch_job(url: str) -> Optional[JobPosting]:

                """Fetch and parse a single job posting."""

                if url in SCRAPE_CACHE:

                    REQUEST_PROGRESS[request_id].jobs_cached += 1

                    REQUEST_PROGRESS[request_id].updated_at = now_iso()

                    cached = SCRAPE_CACHE[url]

                    return JobPosting(url=cached.get('url', url), **{k: v for k, v in cached.items() if k != 'url'})

                

                async with semaphore:

                    try:

                        # Use Firecrawl SDK directly

                        fc = scrape_website_custom(url, settings.firecrawl_api_key)

                        content = ''

                        title = ''

                        company = ''

                        html_content = ''

                        

                        if isinstance(fc, dict) and 'error' not in fc:

                            content = str(fc.get('content') or fc.get('markdown') or fc)

                            md = fc.get('metadata') or {}

                            title = md.get('title') or ''

                            html_content = fc.get('html') or ''



                        # Always parse HTML for better title/company extraction

                        if not requests or not BeautifulSoup:

                            raise ImportError("requests and beautifulsoup4 are required for HTML parsing")

                        

                        if not html_content:

                            headers = {

                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',

                                'Accept-Language': 'en-US,en;q=0.9',

                            }

                            resp = requests.get(url, headers=headers, timeout=20)

                            if resp.ok:

                                html_content = resp.text

                                soup = BeautifulSoup(html_content, 'lxml')

                                

                                # Extract title - try multiple sources

                                if not title:

                                    # Try page title first

                                    if soup.title and soup.title.string:

                                        title = soup.title.string.strip()

                                    

                                    # Try h1 tags (common for job titles)

                                    if not title or len(title) > 100:

                                        h1 = soup.find('h1')

                                        if h1 and h1.get_text(strip=True):

                                            title = h1.get_text(strip=True)

                                    

                                    # Try h2 with common job title classes/ids

                                    if not title or len(title) > 100:

                                        for h2 in soup.find_all('h2', limit=5):

                                            h2_text = h2.get_text(strip=True)

                                            if h2_text and len(h2_text) < 100:

                                                # Check if it looks like a job title

                                                if any(keyword in h2_text.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'executive', 'director', 'assistant', 'coordinator', 'officer']):

                                                    title = h2_text

                                                    break

                                    

                                    # Try meta tags

                                    if not title or len(title) > 100:

                                        og_title = soup.find('meta', property='og:title')

                                        if og_title and og_title.get('content'):

                                            title = og_title.get('content').strip()

                                

                                # Extract company name - try multiple sources

                                if not company:

                                    # Try elements with common class names that contain 'company' (without regex)

                                    def has_company_class(class_attr):

                                        if not class_attr:

                                            return False

                                        if isinstance(class_attr, list):

                                            return any('company' in str(c).lower() for c in class_attr)

                                        return 'company' in str(class_attr).lower()

                                    

                                    # Search common elements

                                    for tag in ['span', 'div', 'a', 'p', 'h3', 'h4']:

                                        elements = soup.find_all(tag, class_=has_company_class)

                                        for elem in elements[:3]:  # Limit per tag type

                                            company_text = elem.get_text(strip=True)

                                            if company_text and 3 <= len(company_text) <= 50:

                                                company = company_text

                                                break

                                        if company:

                                            break

                                    

                                    # Try strong tags with company/employer text

                                    if not company:

                                        strong_tags = soup.find_all('strong')

                                        for strong in strong_tags:

                                            strong_text = strong.get_text(strip=True).lower()

                                            if 'company' in strong_text or 'employer' in strong_text:

                                                # Try to get company name from nearby text

                                                parent = strong.find_parent()

                                                if parent:

                                                    parent_text = parent.get_text(strip=True)

                                                    if len(parent_text) < 100:

                                                        company = parent_text

                                                        break

                                    

                                    # Try meta tags

                                    if not company:

                                        meta_tags = soup.find_all('meta')

                                        for meta in meta_tags:

                                            name_attr = meta.get('name', '').lower()

                                            if name_attr and ('company' in name_attr or 'employer' in name_attr):

                                                content = meta.get('content', '').strip()

                                                if content and 3 <= len(content) <= 50:

                                                    company = content

                                                    break

                                    

                                    # Look in content text for "at [Company]" pattern

                                    if not company and content:

                                        company_match = re.search(r'\bat\s+([A-Z][A-Za-z\s&]{2,40})\b', content[:1000], re.I)

                                        if company_match:

                                            company = company_match.group(1).strip()

                                

                                # Get content if not already extracted

                                if not content:

                                    desc_tag = soup.find('meta', attrs={'name': 'description'})

                                    meta_desc = (desc_tag['content'].strip() if desc_tag and desc_tag.has_attr('content') else '')

                                    main = soup.find('main') or soup.find('body')

                                    text = (main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True))

                                    content = (meta_desc + "\n\n" + text)[:20000]

                        else:

                            # Parse HTML content from Firecrawl

                            soup = BeautifulSoup(html_content, 'lxml')

                            

                            # Extract title - try multiple sources

                            if not title:

                                # Try page title first

                                if soup.title and soup.title.string:

                                    title = soup.title.string.strip()

                                

                                # Try h1 tags

                                if not title or len(title) > 100:

                                    h1 = soup.find('h1')

                                    if h1 and h1.get_text(strip=True):

                                        title = h1.get_text(strip=True)

                                

                                # Try common job title selectors

                                job_title_selectors = [

                                    'h1.job-title', 'h2.job-title', '.job-title', 

                                    '[data-testid*="job-title"]', '[class*="JobTitle"]',

                                    'h1', 'h2'

                                ]

                                for selector in job_title_selectors:

                                    elem = soup.select_one(selector)

                                    if elem:

                                        title_text = elem.get_text(strip=True)

                                        if title_text and len(title_text) < 100:

                                            title = title_text

                                            break

                            

                            # Extract company name

                            if not company:

                                company_selectors = [

                                    '.company-name', '[class*="Company"]', 

                                    '[data-testid*="company"]', 'a[href*="/company/"]',

                                    'strong', '.employer'

                                ]

                                for selector in company_selectors:

                                    elem = soup.select_one(selector)

                                    if elem:

                                        company_text = elem.get_text(strip=True)

                                        if company_text and 3 <= len(company_text) <= 50:

                                            company = company_text

                                            break

                            

                            # Ensure content is extracted if not already

                            if not content:

                                desc_tag = soup.find('meta', attrs={'name': 'description'})

                                meta_desc = (desc_tag['content'].strip() if desc_tag and desc_tag.has_attr('content') else '')

                                main = soup.find('main') or soup.find('body')

                                text = (main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True))

                                content = (meta_desc + "\n\n" + text)[:20000]



                        # Clean up extracted values using our cleaning functions
                        # First try to use provided titles/companies from request
                        fallback_title = job_titles.get(url, '') if url in job_titles else title
                        fallback_company = job_companies.get(url, '') if url in job_companies else company
                        
                        # Clean the extracted values
                        title = clean_job_title(title) or clean_job_title(fallback_title)
                        company = clean_company_name(company) or clean_company_name(fallback_company)
                        
                        # If title/company not found, try extracting from content
                        if not title:
                            title = extract_job_title_from_content(content, fallback_title)
                        
                        if not company:
                            company = extract_company_name_from_content(content, fallback_company)


                        print(f"\n✓ Scraped {url} ({len(content)} chars)")

                        if title:

                            print(f"  Title extracted: {title}")

                        if company:

                            print(f"  Company extracted: {company}")



                        # Ensure we have valid title and company (use fallback messages if not found)
                        final_title = title or "Job title not available in posting"
                        final_company = company or "Company name not available in posting"
                        
                        print(f"[Job Info] Final job title: {final_title}")
                        print(f"[Job Info] Final company: {final_company}")
                        

                        job = JobPosting(

                            url=url,

                            description=content,

                            job_title=final_title,

                            company=final_company,

                        )

                        

                        # Cache

                        cache_data = job.dict()

                        cache_data['url'] = str(cache_data['url'])

                        cache_data['scraped_summary'] = content[:200] + "..." if len(content) > 200 else content

                        SCRAPE_CACHE[url] = cache_data

                        

                        REQUEST_PROGRESS[request_id].jobs_scraped += 1

                        REQUEST_PROGRESS[request_id].updated_at = now_iso()

                        return job

                    

                    except Exception as e:

                        print(f"❌ Error scraping {url}: {e}")

                        return None



            jobs: List[JobPosting] = [

                j for j in await asyncio.gather(*[fetch_job(u) for u in urls]) 

                if j is not None

            ]



            if not jobs:

                raise HTTPException(status_code=500, detail="Failed to scrape any job postings")



        # STEP 4: Extract company names early (for parallel sponsorship checking)
        company_names_map = {}  # Map job URL to company name
        for job in jobs:
            if job.company and job.company not in ["Company name not available in posting", "Not specified"]:
                company_names_map[str(job.url)] = job.company
        
        # STEP 5: Run Job Scoring and Sponsorship Checking in PARALLEL
        REQUEST_PROGRESS[request_id].status = "matching"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        
        logger.info("Starting parallel execution: job scoring and sponsorship checking")
        
        scorer_agent = build_scorer(settings.model_name)



        def score_job_sync(job: JobPosting) -> Optional[Dict[str, Any]]:

            """Score a single job using AI reasoning."""

            try:

                prompt = f"""

Analyze the match between candidate and job. Consider ALL requirements from the job description.

Candidate Profile:

{json.dumps(candidate_profile.dict(), indent=2)}

Job Details:

- Title: {job.job_title}

- Company: {job.company}

- URL: {str(job.url)}

- Description: {job.description[:2000]}

CRITICAL: Read the job description carefully. If this is a:

- Billing/Finance role: Score based on financial/accounting skills

- Tech/Engineering role: Score based on technical skills

- Sales/Marketing role: Score based on communication/business skills

Return ONLY valid JSON (no markdown) with the following structure:

{{
  "match_score": 0.75,
  "key_matches": ["skill1", "skill2"],
  "requirements_met": 5,
  "total_requirements": 8,
  "requirements_satisfied": [
    "Java (candidate has 3 years experience)",
    "React (candidate has strong frontend experience)",
    "AWS (candidate has cloud experience)"
  ],
  "requirements_missing": [
    "Kubernetes (not mentioned in candidate profile)",
    "Docker (not mentioned in candidate profile)",
    "5+ years experience (candidate has 3 years)"
  ],
  "improvements_needed": [
    "Gain experience with containerization tools (Docker, Kubernetes)",
    "Build 2 more years of experience to meet the 5+ years requirement",
    "Learn CI/CD pipeline tools (Jenkins, GitLab CI)"
  ],
  "reasoning": "Brief explanation of score"
}}

IMPORTANT INSTRUCTIONS:

1. **requirements_satisfied**: List ALL specific requirements/skills from the job that the candidate MATCHES.
   - Include the requirement name and brief context (e.g., "Java (candidate has 3 years experience)")
   - Be specific: "Java, Spring Boot" not just "Java"
   - Include experience matches: "5+ years (candidate has 4 years)" if close
   - Include technology matches: "React (candidate has strong frontend experience)"
   - Include soft skills if mentioned: "Team leadership (candidate has managed teams)"

2. **requirements_missing**: List ALL specific requirements/skills from the job that the candidate DOES NOT MATCH.
   - Include the requirement name and why it's missing (e.g., "Kubernetes (not mentioned in candidate profile)")
   - Be specific about what's missing
   - Include experience gaps: "5+ years experience (candidate has 3 years)"
   - Include missing technologies, tools, certifications, etc.

3. **improvements_needed**: List SPECIFIC, ACTIONABLE improvements the candidate should work on.
   - Focus on the most important missing requirements
   - Be specific: "Learn Docker and Kubernetes" not just "Learn containerization"
   - Include experience gaps: "Gain 2 more years of experience"
   - Prioritize improvements that would have the biggest impact on match score
   - Keep it practical and achievable

4. **requirements_met** and **total_requirements**: Count the number of requirements satisfied vs total requirements.
   - Count each distinct requirement (skill, tool, experience level, certification, etc.)
   - Be generous but accurate - count related skills as matches if the candidate has similar experience

5. **match_score**: Calculate based on:
   - Percentage of requirements satisfied
   - Importance of satisfied/missing requirements
   - Experience level match
   - Overall fit quality

Be strict with scoring:

- < 0.3: Poor fit (major skill gaps)

- 0.3-0.5: Weak fit (some alignment)

- 0.5-0.7: Good fit (strong alignment)

- > 0.7: Excellent fit (ideal candidate)

"""

                # Use OpenAI streaming API directly for better control
                openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)
                        
                        # Use streaming to get response faster (but collect full response)
                        # Note: Some models don't support temperature=0, so we omit it to use default
                        stream = client.chat.completions.create(
                            model=settings.model_name or "gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            stream=True
                        )
                        
                        # Collect streaming response
                        response_text = ""
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                response_text += chunk.choices[0].delta.content
                        
                        # Create a mock response object for compatibility
                        class MockResponse:
                            def __init__(self, content):
                                self.content = content
                        response = MockResponse(response_text)
                    except Exception as e:
                        logger.warning(f"OpenAI streaming failed, falling back to agent: {e}")
                        response = scorer_agent.run(prompt)
                else:
                    response = scorer_agent.run(prompt)

                

                # Handle different response types

                if hasattr(response, 'content'):

                    response_text = str(response.content)

                elif hasattr(response, 'messages') and response.messages:

                    # Get last message content

                    last_msg = response.messages[-1]

                    response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)

                else:

                    response_text = str(response)

                

                response_text = response_text.strip()

                

                print(f"\n[SCORER RAW OUTPUT for {job.job_title}]:")

                print(response_text[:500])

                

                # Extract JSON from response

                data = extract_json_from_response(response_text)

                

                # Validate and extract score with defaults

                if not data or data.get("match_score") is None:

                    print(f"⚠️  Warning: Could not extract match_score from response, using default 0.5")

                    data = data or {}

                    data["match_score"] = 0.5

                

                score = float(data.get("match_score", 0.5))

                print(f"✓ Scored {job.job_title}: {score:.1%}")

                # Fix requirements calculation: if total_requirements is 0, calculate from satisfied + missing
                requirements_satisfied_list = data.get("requirements_satisfied", []) or []
                requirements_missing_list = data.get("requirements_missing", []) or []
                requirements_met = int(data.get("requirements_met", 0))
                total_requirements = int(data.get("total_requirements", 0))
                
                # If total_requirements is 0 but we have satisfied/missing lists, calculate from them
                if total_requirements == 0:
                    total_requirements = len(requirements_satisfied_list) + len(requirements_missing_list)
                    if total_requirements == 0:
                        total_requirements = 1  # Avoid division by zero
                    requirements_met = len(requirements_satisfied_list)
                
                # Ensure requirements_met doesn't exceed total_requirements
                if requirements_met > total_requirements:
                    requirements_met = total_requirements

                return {

                    "job": job,

                    "match_score": score,

                    "key_matches": data.get("key_matches", []) or [],

                    "requirements_met": requirements_met,

                    "total_requirements": total_requirements,

                    "requirements_satisfied": requirements_satisfied_list,

                    "requirements_missing": requirements_missing_list,

                    "improvements_needed": data.get("improvements_needed", []) or [],

                    "reasoning": data.get("reasoning", "Score calculated based on candidate-job alignment"),

                }

            except Exception as e:

                print(f"❌ Error scoring {job.job_title}: {e}")

                return None



        # Score jobs in parallel with semaphore for rate limiting
        scoring_semaphore = asyncio.Semaphore(10)  # Allow up to 10 concurrent scoring operations for faster processing
        
        async def score_job_async(job: JobPosting) -> Optional[Dict[str, Any]]:
            """Score a single job asynchronously with rate limiting."""
            async with scoring_semaphore:
                # Run the synchronous scoring function in a thread pool
                result = await asyncio.to_thread(score_job_sync, job)
                # Removed delay for faster processing
                return result
        
        # Helper function to check if visa details are already in job content
        def has_visa_details_in_content(content: str) -> bool:
            """Check if visa/sponsorship details are already mentioned in the job content."""
            if not content:
                return False
            
            content_lower = content.lower()
            visa_keywords = [
                "visa sponsorship", "visa support", "sponsor visa", "visa sponsor",
                "work permit", "visa assistance", "sponsorship available",
                "uk visa sponsor", "registered sponsor", "tier 2", "tier 5",
                "skilled worker visa", "sponsor license", "sponsor licence"
            ]
            
            return any(keyword in content_lower for keyword in visa_keywords)
        
        # Define async function for sponsorship checking (can run in parallel with scoring)
        async def check_sponsorship_for_companies_async() -> Optional[Dict[str, Any]]:
            """Check sponsorship for all company names in parallel."""
            if not company_names_map:
                return None
            
            try:
                from sponsorship_checker import check_sponsorship
                sponsorship_semaphore = asyncio.Semaphore(3)
                
                async def check_single_company(company_name: str, job_url: str) -> Optional[Dict[str, Any]]:
                    """Check sponsorship for a single company."""
                    async with sponsorship_semaphore:
                        try:
                            # Get job content from cache if available
                            job_content = ""
                            if job_url:
                                cached_data = SCRAPE_CACHE.get(job_url, {})
                                job_content = cached_data.get('description') or cached_data.get('scraped_summary') or cached_data.get('text_content') or ""
                            
                            # Check if visa details are already in the job content
                            # If they are, skip the sponsorship check
                            if has_visa_details_in_content(job_content):
                                logger.debug(f"Skipping sponsorship check for {company_name} - visa details already in job content")
                                return None
                            
                            # Also check in job description from jobs list
                            job_obj = next((j for j in jobs if str(j.url) == job_url), None)
                            if job_obj and job_obj.description:
                                if has_visa_details_in_content(job_obj.description):
                                    logger.debug(f"Skipping sponsorship check for {company_name} - visa details already in job description")
                                    return None
                            
                            # Clean company name
                            cleaned_name = clean_company_name(company_name)
                            if not cleaned_name or not is_valid_company_name(cleaned_name):
                                return None
                            
                            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                            sponsorship_result = await asyncio.to_thread(check_sponsorship, cleaned_name, job_content, openai_key)
                            
                            return {
                                'company_name': sponsorship_result.get('company_name'),
                                'sponsors_workers': sponsorship_result.get('sponsors_workers', False),
                                'visa_types': sponsorship_result.get('visa_types'),
                                'summary': sponsorship_result.get('summary', 'No sponsorship information available'),
                                'job_url': job_url
                            }
                        except Exception as e:
                            logger.error(f"Error checking sponsorship for {company_name}: {e}", exc_info=True)
                            return None
                
                # Check all companies in parallel
                sponsorship_tasks = [
                    check_single_company(company_name, job_url)
                    for job_url, company_name in company_names_map.items()
                ]
                sponsorship_results = await asyncio.gather(*sponsorship_tasks, return_exceptions=True)
                
                # Filter valid results
                valid_results = [r for r in sponsorship_results if r and not isinstance(r, Exception)]
                if valid_results:
                    # Return the first valid result (prioritize by job order later)
                    return valid_results[0]
                return None
            except Exception as e:
                logger.error(f"Error in parallel sponsorship checking: {e}", exc_info=True)
                return None
        
        # Run scoring and sponsorship checking in PARALLEL
        logger.info(f"Scoring {len(jobs)} jobs and checking sponsorship for {len(company_names_map)} companies in parallel")
        scoring_tasks = [score_job_async(job) for job in jobs]
        
        # Execute both in parallel
        scoring_results, sponsorship_result = await asyncio.gather(
            asyncio.gather(*scoring_tasks, return_exceptions=True),
            check_sponsorship_for_companies_async(),
            return_exceptions=True
        )
        
        # Handle sponsorship result
        if isinstance(sponsorship_result, Exception):
            logger.warning(f"Sponsorship checking failed: {sponsorship_result}")
            sponsorship_result = None
        
        # Filter out exceptions and None results from scoring
        scored = []
        for result in scoring_results:
            if isinstance(result, Exception):
                logger.error(f"Error in parallel scoring: {result}")
                continue
            if result:
                scored.append(result)



        # Sort by match score and take top matches

        scored.sort(key=lambda x: x["match_score"], reverse=True)

        

        # Only summarize jobs with decent match scores

        top_matches = [s for s in scored if s["match_score"] >= 0.5][:10]

        

        if not top_matches:

            print("⚠️  No jobs with match score >= 50%, taking top 5")

            top_matches = scored[:5]



        # STEP 5: Generate Summaries

        REQUEST_PROGRESS[request_id].status = "summarizing"

        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        

        print("\n" + "="*80)

        print(f"📝 SUMMARIZER AGENT - Generating summaries for {len(top_matches)} jobs")

        print("="*80)



        summarizer_agent = build_summarizer(settings.model_name)



        def summarize_sync(entry: Dict[str, Any], rank: int) -> MatchedJob:

            """Generate summary for a matched job using the summarizer agent."""

            job: JobPosting = entry["job"]

            score = entry["match_score"]

            scoring_reasoning = entry.get("reasoning", "")
            requirements_satisfied = entry.get("requirements_satisfied", [])[:3]
            requirements_missing = entry.get("requirements_missing", [])[:3]
            key_matches = entry.get("key_matches", [])[:5]
            
            # Build simple prompt for summarizer agent
            summary_prompt = f"""Create a concise analysis summary (under 500 characters) for this job match.

Match Score: {score:.1%}
Job Title: {job.job_title}
Company: {job.company}
Candidate: {candidate_profile.name}

Scoring Reasoning: {scoring_reasoning[:300] if scoring_reasoning else "N/A"}

Key Matching Skills: {', '.join(key_matches) if key_matches else "N/A"}

Requirements Satisfied:
{chr(10).join(f"- {req}" for req in requirements_satisfied[:2]) if requirements_satisfied else "N/A"}

Requirements Missing:
{chr(10).join(f"- {req}" for req in requirements_missing[:2]) if requirements_missing else "N/A"}

Job Description:
{job.description[:1000] if job.description else "N/A"}

Generate a concise summary that:
1. States the match score and fit assessment
2. Explains key matching skills
3. Highlights areas that need improvement
4. Keep it under 500 characters and end at a complete sentence."""

            text = ""
            try:
                # Use the summarizer agent (simple method)
                summary_response = summarizer_agent.run(summary_prompt)
                
                # Extract response content
                if hasattr(summary_response, "content"):
                    text = str(summary_response.content).strip()
                elif hasattr(summary_response, "messages") and summary_response.messages:
                    last_msg = summary_response.messages[-1]
                    text = str(last_msg.content if hasattr(last_msg, "content") else last_msg).strip()
                else:
                    text = str(summary_response).strip()
                
                # Clean markdown formatting if present
                text = clean_summary_text(text)
                
                # Strip markdown code fences if present
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1])
                    text = text.strip()
                
                # Truncate to 500 characters at last complete sentence
                if len(text) > 500:
                    truncated = text[:500]
                    # Find last complete sentence
                    last_period = truncated.rfind('.')
                    last_exclamation = truncated.rfind('!')
                    last_question = truncated.rfind('?')
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    
                    if last_sentence_end > 400:  # At least 80% of max
                        text = text[:last_sentence_end + 1].strip()
                    else:
                        # Find word boundary (NEVER cut mid-word)
                        last_space = truncated.rfind(' ')
                        if last_space > 400:
                            text = text[:last_space].strip() + "."
                        else:
                            # Find any space in the last 100 chars
                            search_start = max(0, 500 - 100)
                            last_space_fallback = truncated.rfind(' ', search_start)
                            if last_space_fallback > 350:
                                text = text[:last_space_fallback].strip() + "."
                            else:
                                # Find ANY space
                                any_space = truncated.rfind(' ')
                                if any_space > 0:
                                    text = text[:any_space].strip() + "."
                                else:
                                    text = truncated.strip() + "."
                
                # Ensure it ends with proper punctuation
                if text and text[-1] not in ['.', '!', '?']:
                    text = text.rstrip() + "."
                
                logger.debug(f"✓ Summarized rank {rank}: {job.job_title} ({len(text)} chars)")

            except Exception as e:
                logger.error(f"❌ Error summarizing rank {rank}: {e}")
                # Fallback to simple summary
                text = f"Match score: {score:.1%}. {scoring_reasoning[:300] if scoring_reasoning else 'Score calculated based on candidate-job alignment'}."
                if len(text) > 500:
                    truncated = text[:500]
                    last_period = truncated.rfind('.')
                    if last_period > 400:
                        text = text[:last_period + 1].strip()
                    else:
                        last_space = truncated.rfind(' ')
                        if last_space > 400:
                            text = text[:last_space].strip() + "."
                        else:
                            text = truncated.strip() + "."

            

            # Extract visa/scholarship information from job description

            visa_scholarship_info = None

            job_description_lower = (job.description or "").lower()

            job_text = job_description_lower

            

            # Check for visa/scholarship keywords

            visa_keywords = ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit", "financial support", "tuition assistance", "visa assistance"]

            found_keywords = [kw for kw in visa_keywords if kw in job_text]

            

            if found_keywords:

                # Extract context around the keyword

                for keyword in found_keywords:

                    idx = job_text.find(keyword)

                    if idx != -1:

                        # Get surrounding context (100 chars before, 200 chars after)

                        start = max(0, idx - 100)

                        end = min(len(job.description or ""), idx + len(keyword) + 200)

                        context = (job.description or "")[start:end].strip()

                        visa_scholarship_info = context

                        break

            else:

                # Check in scraped cache if available

                cached_data = SCRAPE_CACHE.get(str(job.url), {})

                if cached_data.get("summarized_data", {}).get("visa_scholarship_info"):

                    visa_scholarship_info = cached_data["summarized_data"]["visa_scholarship_info"]

                elif cached_data.get("scraped_data", {}).get("text_content"):

                    cached_text = cached_data["scraped_data"]["text_content"].lower()

                    for keyword in visa_keywords:

                        if keyword in cached_text:

                            idx = cached_text.find(keyword)

                            start = max(0, idx - 100)

                            end = min(len(cached_data["scraped_data"]["text_content"]), idx + len(keyword) + 200)

                            context = cached_data["scraped_data"]["text_content"][start:end].strip()

                            visa_scholarship_info = context

                            break

            

            if not visa_scholarship_info:

                visa_scholarship_info = "Not specified"

            

            # Extract location from job description
            location = None
            if job.description:
                try:
                    from sponsorship_checker import _extract_location_from_job_content
                    location = _extract_location_from_job_content(job.description)
                    if location:
                        print(f"[Location] Extracted location for job {rank}: {location}")
                except Exception as e:
                    print(f"[Location] Error extracting location: {e}")
            
            return MatchedJob(

                rank=rank,

                job_url=str(job.url),

                job_title=job.job_title or "Unknown",

                company=job.company or "Unknown",

                match_score=round(score, 3),

                summary=text,

                key_matches=entry["key_matches"],

                requirements_met=entry["requirements_met"],

                total_requirements=entry["total_requirements"],

                requirements_satisfied=entry.get("requirements_satisfied", []) or [],

                requirements_missing=entry.get("requirements_missing", []) or [],

                improvements_needed=entry.get("improvements_needed", []) or [],

                location=location,

                scraped_summary=None,  # Remove duplicate - summary field contains all needed info

            )



        # Generate summaries in parallel with semaphore for rate limiting
        summarization_semaphore = asyncio.Semaphore(5)  # Allow up to 5 concurrent summarizations for faster processing
        
        async def summarize_async(entry: Dict[str, Any], rank: int) -> MatchedJob:
            """Generate summary for a matched job asynchronously with rate limiting."""
            async with summarization_semaphore:
                # Run the synchronous summarization function in a thread pool
                result = await asyncio.to_thread(summarize_sync, entry, rank)
                # Removed delay for faster processing
                return result
        
        # Generate summaries in parallel
        print(f"[SUMMARIZATION] Starting parallel summarization for {len(top_matches)} jobs...")
        summarization_tasks = [summarize_async(entry, i + 1) for i, entry in enumerate(top_matches)]
        matched_jobs = await asyncio.gather(*summarization_tasks, return_exceptions=True)
        
        # Filter out exceptions
        matched_jobs = [job for job in matched_jobs if not isinstance(job, Exception)]



        print("\n" + "="*80)

        print("✅ FINAL RESPONSE - All agents completed")

        print("="*80)

        print(f"Found {len(matched_jobs)} matched jobs out of {len(jobs)} analyzed")

        print(f"Top match: {matched_jobs[0].job_title} ({matched_jobs[0].match_score:.1%})")

        print(f"Request ID: {request_id}")

        print("="*80 + "\n")



        REQUEST_PROGRESS[request_id].status = "completed"

        REQUEST_PROGRESS[request_id].updated_at = now_iso()



        # Get user_id from request (already extracted earlier for new_format_jobs at line 715)

        if not user_id:

            if data:

                user_id = data.user_id

            elif legacy_data:

                user_id = legacy_data.user_id



        # Schedule Firebase save in background if user_id is provided (user doesn't wait)
        if user_id and matched_jobs:
            try:
                from job_extractor import extract_jobs_from_response
                
                # Convert MatchedJob Pydantic objects to dictionaries for extraction function
                api_response_format = {
                    "matched_jobs": [
                        {
                            "rank": job.rank,
                            "job_url": str(job.job_url),
                            "job_title": job.job_title,
                            "company": job.company,
                            "match_score": job.match_score,
                            "summary": job.summary,
                            "key_matches": job.key_matches,
                            "requirements_met": job.requirements_met,
                            "total_requirements": job.total_requirements,
                            "location": job.location,
                        }
                        for job in matched_jobs
                    ]
                }
                
                jobs_to_save = extract_jobs_from_response(api_response_format)
                
                # Schedule Firebase save in background (user doesn't wait)
                if jobs_to_save:
                    logger.info(f"Scheduling background save of {len(jobs_to_save)} job applications")
                    background_tasks.add_task(
                        save_job_applications_background,
                        user_id,
                        jobs_to_save
                    )
                else:
                    logger.debug("No job applications to save")
            except Exception as e:
                logger.error(f"Error preparing background save: {e}", exc_info=True)
                # Non-fatal - continue with response



        # STEP 6: Process sponsorship result (already checked in parallel above)
        sponsorship_info = None
        if sponsorship_result:
            try:
                from models import SponsorshipInfo
                
                sponsorship_info = SponsorshipInfo(
                    company_name=sponsorship_result.get('company_name'),
                    sponsors_workers=sponsorship_result.get('sponsors_workers', False),
                    visa_types=sponsorship_result.get('visa_types'),
                    summary=sponsorship_result.get('summary', 'No sponsorship information available')
                    # Note: document_id and document_data are NOT included in API response
                )
                
                logger.info(f"Sponsorship result: {'Sponsors workers' if sponsorship_info.sponsors_workers else 'Does not sponsor workers'}")
                if sponsorship_info.visa_types:
                    logger.debug(f"Visa types: {sponsorship_info.visa_types}")
            except Exception as e:
                logger.error(f"Error creating SponsorshipInfo: {e}", exc_info=True)
                sponsorship_info = None
            
            # Update matched job summary to reflect actual sponsorship info (remove "No mention..." text)
            if matched_jobs and len(matched_jobs) > 0 and sponsorship_info:
                top_job = matched_jobs[0]
                if top_job.summary:
                    summary_text = top_job.summary
                    
                    # Remove "No mention..." or similar text about sponsorship
                    patterns_to_remove = [
                        r'No\s+specific\s+information\s+about\s+visa\s+sponsorship[^.]*\.',
                        r'No\s+mention\s+was\s+made\s+of\s+any\s+visa\s+sponsorship[^.]*\.',
                        r'No\s+(?:mention\s+was\s+made\s+of\s+any\s+)?visa\s+sponsorship[^.]*\.',
                        r'No\s+visa\s+sponsorship[^.]*\.',
                        r'visa\s+sponsorship[^.]*not\s+mentioned[^.]*\.',
                        r'visa\s+sponsorship[^.]*is\s+not\s+mentioned[^.]*\.',
                        r'no\s+information\s+about\s+visa\s+sponsorship[^.]*\.',
                        r'scholarship[^.]*not\s+mentioned[^.]*\.',
                    ]
                    
                    for pattern in patterns_to_remove:
                        summary_text = re.sub(pattern, '', summary_text, flags=re.IGNORECASE)
                    
                    # Normalize whitespace
                    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                    summary_text = re.sub(r'\s+([,.;])\s+', r'\1 ', summary_text)
                    summary_text = re.sub(r'\s+([,.;])\s*$', '', summary_text)
                    
                    # Remove any sponsorship-related text from summary (sponsorship info goes to separate field)
                    # Remove patterns that might have been added by previous logic
                    sponsorship_patterns_to_remove = [
                        r'Visa\s+Sponsorship[^.]*\.',
                        r'visa\s+sponsor[^.]*\.',
                        r'registered\s+UK\s+visa\s+sponsor[^.]*\.',
                        r'Visa\s+Routes[^.]*\.',
                    ]
                    
                    for pattern in sponsorship_patterns_to_remove:
                        summary_text = re.sub(pattern, '', summary_text, flags=re.IGNORECASE)
                    
                    # Normalize whitespace again after removal
                    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                    summary_text = re.sub(r'\s+([,.;])\s+', r'\1 ', summary_text)
                    summary_text = re.sub(r'\s+([,.;])\s*$', '', summary_text)
                    
                    summary_text = clean_summary_text(summary_text)
                    top_job.summary = summary_text
                    logger.debug("Removed sponsorship details from summary (sponsorship info is in separate field)")
            
            # Schedule sponsorship save in background
            if sponsorship_info and user_id:
                try:
                    sponsorship_dict = {
                        "company_name": sponsorship_info.company_name,
                        "sponsors_workers": sponsorship_info.sponsors_workers,
                        "visa_types": sponsorship_info.visa_types,
                        "summary": sponsorship_info.summary
                    }
                    
                    job_info = None
                    if matched_jobs and len(matched_jobs) > 0:
                        top_job = matched_jobs[0]
                        portal = "Unknown"
                        job_url_str = str(top_job.job_url) if top_job.job_url else ""
                        if "linkedin.com" in job_url_str.lower():
                            portal = "LinkedIn"
                        elif "indeed.com" in job_url_str.lower():
                            portal = "Indeed"
                        elif "glassdoor.com" in job_url_str.lower():
                            portal = "Glassdoor"
                        
                        job_info = {
                            "job_title": top_job.job_title,
                            "job_url": job_url_str,
                            "company": top_job.company,
                            "portal": portal
                        }
                    
                    logger.info(f"Scheduling background save of sponsorship info for {sponsorship_dict.get('company_name')}")
                    background_tasks.add_task(
                        save_sponsorship_info_background,
                        user_id,
                        request_id,
                        sponsorship_dict,
                        job_info
                    )
                except Exception as e:
                    logger.error(f"Error scheduling sponsorship save: {e}", exc_info=True)

        response = MatchJobsResponse(

            candidate_profile=CandidateProfile(
                name=candidate_profile.name,
                email=candidate_profile.email,
                phone=candidate_profile.phone,
                skills=candidate_profile.skills,
                experience_summary=candidate_profile.experience_summary,
                total_years_experience=candidate_profile.total_years_experience,
                interests=candidate_profile.interests,
                education=candidate_profile.education,
                certifications=candidate_profile.certifications,
                # Note: raw_text_excerpt is NOT included (internal field)
            ),

            matched_jobs=matched_jobs,

            processing_time=f"{time.time() - start_time:.1f}s",

            jobs_analyzed=len(jobs),  # Use len(jobs) instead of len(urls) for accuracy

            request_id=request_id,

            sponsorship=SponsorshipInfo(
                company_name=sponsorship_info.company_name,
                sponsors_workers=sponsorship_info.sponsors_workers,
                visa_types=sponsorship_info.visa_types,
                summary=sponsorship_info.summary,
                # Note: document_id and document_data are NOT included (internal fields)
            ) if sponsorship_info else None,
        )

        return response

        

    except HTTPException:

        REQUEST_PROGRESS[request_id].status = "error"

        REQUEST_PROGRESS[request_id].error = "HTTP error"

        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        raise

    except Exception as e:

        REQUEST_PROGRESS[request_id].status = "error"

        REQUEST_PROGRESS[request_id].error = str(e)

        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        import traceback

        print(f"Full error traceback: {traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





@app.post("/api/match-jobs/stream")
async def match_jobs_stream(
    json_body: Optional[str] = Form(default=None),
    pdf_file: Optional[UploadFile] = File(default=None),
    settings: Settings = Depends(get_settings),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Streaming version of match-jobs endpoint.
    Streams scoring progress as Server-Sent Events (SSE) as the LLM generates responses.
    Returns the same data as /api/match-jobs but streams progress updates.
    """
    # CRITICAL: Read file contents into memory BEFORE starting the stream
    # The UploadFile object gets closed after the request handler returns,
    # so we must read it before passing to the generator
    resume_bytes_from_file: Optional[bytes] = None
    if pdf_file is not None:
        try:
            resume_bytes_from_file = await pdf_file.read()
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}", exc_info=True)
            return StreamingResponse(
                content=f"data: {json.dumps({'type': 'error', 'error': f'Failed to read PDF file: {str(e)}'})}\n\n",
                media_type="text/event-stream"
            )
    
    async def generate_stream(resume_bytes_preloaded: Optional[bytes] = None):
        try:
            from openai import OpenAI
            
            request_id = make_request_id()
            start_time = time.time()
            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            
            if not openai_key:
                yield f"data: {json.dumps({'type': 'error', 'error': 'OpenAI API key not configured'})}\n\n"
                return
            
            client = OpenAI(api_key=openai_key)
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'status': 'parsing', 'request_id': request_id, 'message': 'Parsing request...'})}\n\n"
            
            # Parse request (reuse logic from match_jobs)
            # This is a simplified version - in production, extract the full parsing logic
            data: Optional[MatchJobsRequest] = None
            legacy_data: Optional[MatchJobsJsonRequest] = None
            new_format_jobs: Optional[Dict[str, Any]] = None
            jobs_string: Optional[str] = None
            user_id: Optional[str] = None
            
            if json_body:
                clean_json = json_body.strip()
                if clean_json.startswith('"') and clean_json.endswith('"'):
                    clean_json = clean_json[1:-1].replace('\\"', '"')
                payload = json.loads(clean_json)
                
                if "jobs" in payload and isinstance(payload["jobs"], dict):
                    new_format_jobs = payload["jobs"]
                    user_id = payload.get("user_id")
                elif "jobs" in payload and isinstance(payload["jobs"], str):
                    jobs_string = payload["jobs"]
                    user_id = payload.get("user_id")
                elif "resume" in payload and "jobs" in payload:
                    data = MatchJobsRequest(**payload)
                else:
                    try:
                        legacy_data = MatchJobsJsonRequest(**payload)
                    except:
                        yield f"data: {json.dumps({'type': 'error', 'error': 'Invalid request format'})}\n\n"
                        return
            
            # Get resume - use preloaded bytes if available
            resume_bytes: Optional[bytes] = None
            if data and data.resume and data.resume.content:
                resume_bytes = decode_base64_pdf(data.resume.content)
            elif legacy_data and legacy_data.pdf:
                resume_bytes = decode_base64_pdf(legacy_data.pdf)
            elif resume_bytes_preloaded is not None:
                resume_bytes = resume_bytes_preloaded
            
            if not resume_bytes:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Missing resume PDF'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'status': 'parsing_resume', 'message': 'Parsing resume...'})}\n\n"
            
            # Parse resume
            resume_text = extract_text_from_pdf_bytes(resume_bytes)
            from resume_parser_ocr import parse_resume_with_llm_fallback
            model_name = settings.model_name or "gpt-4o-mini"
            candidate_profile = await asyncio.to_thread(parse_resume_with_llm_fallback, resume_text, model_name, openai_key)
            
            yield f"data: {json.dumps({'type': 'status', 'status': 'extracting_jobs', 'message': 'Extracting job information...'})}\n\n"
            
            # Extract jobs (simplified - handle new_format_jobs case)
            jobs = []
            if new_format_jobs:
                raw_job_title = new_format_jobs.get("jobtitle", "")
                job_link = new_format_jobs.get("joblink", "")
                job_data = new_format_jobs.get("jobdata", "")
                
                from job_extractor import extract_company_and_title_from_raw_data
                extracted_info = await asyncio.to_thread(
                    extract_company_and_title_from_raw_data,
                    job_data,
                    openai_key,
                    "gpt-4o-mini"
                )
                
                extracted_company = extracted_info.get("company_name")
                extracted_title = extracted_info.get("job_title")
                final_job_title = extracted_title or raw_job_title or "Job title not available in posting"
                final_company = extracted_company or "Company name not available in posting"
                
                if final_job_title and final_job_title != "Job title not available in posting":
                    cleaned_title = clean_job_title(final_job_title)
                    if cleaned_title and len(cleaned_title) >= 5:
                        final_job_title = cleaned_title
                    else:
                        final_job_title = raw_job_title.strip() if raw_job_title else "Job title not available in posting"
                
                job = JobPosting(
                    url=job_link if job_link else "https://example.com",
                    job_title=final_job_title,
                    company=final_company,
                    description=job_data,
                    skills_needed=[],
                    experience_level=None,
                    salary=None
                )
                jobs = [job]
            
            if not jobs:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No jobs found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'status': 'scoring', 'message': f'Scoring {len(jobs)} job(s)...', 'jobs_count': len(jobs)})}\n\n"
            
            # Stream scoring for each job
            scored_jobs = []
            for idx, job in enumerate(jobs, 1):
                yield f"data: {json.dumps({'type': 'job_start', 'job_index': idx, 'total_jobs': len(jobs), 'job_title': job.job_title, 'company': job.company})}\n\n"
                
                # Create scoring prompt
                prompt = f"""
Analyze the match between candidate and job. Consider ALL requirements from the job description.

Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Job Details:
- Title: {job.job_title}
- Company: {job.company}
- URL: {str(job.url)}
- Description: {job.description[:2000]}

Return ONLY valid JSON (no markdown) with the following structure:
{{
  "match_score": 0.75,
  "key_matches": ["skill1", "skill2"],
  "requirements_met": 5,
  "total_requirements": 8,
  "requirements_satisfied": ["Java (candidate has 3 years experience)"],
  "requirements_missing": ["Kubernetes (not mentioned in candidate profile)"],
  "improvements_needed": ["Learn Docker and Kubernetes"],
  "reasoning": "Brief explanation of score"
}}
"""
                
                # Stream the scoring response
                # Note: Some models don't support temperature=0, so we omit it to use default
                full_response = ""
                
                # Use a simple queue.Queue for thread-safe communication
                import queue
                chunk_queue = queue.Queue()
                stream_done = False
                stream_error = None
                
                def stream_openai_response():
                    """Run OpenAI streaming in a thread and put chunks in queue"""
                    nonlocal stream_done, stream_error
                    try:
                        logger.info(f"Starting OpenAI stream for job {idx}")
                        stream = client.chat.completions.create(
                            model=settings.model_name or "gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            stream=True
                        )
                        chunk_count = 0
                        for chunk in stream:
                            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                chunk_queue.put(content)
                                chunk_count += 1
                        logger.info(f"OpenAI stream completed, {chunk_count} chunks received")
                        chunk_queue.put(None)  # Signal completion
                    except Exception as e:
                        logger.error(f"Error in OpenAI stream: {e}", exc_info=True)
                        stream_error = str(e)
                        chunk_queue.put(("error", str(e)))
                    finally:
                        stream_done = True
                
                # Start streaming in a thread
                loop = asyncio.get_event_loop()
                stream_task = loop.run_in_executor(None, stream_openai_response)
                
                # Yield chunks from queue as they arrive (using asyncio.to_thread for non-blocking get)
                try:
                    while True:
                        try:
                            # Use asyncio.to_thread to make queue.get() non-blocking
                            def get_chunk():
                                try:
                                    return chunk_queue.get(timeout=0.1)
                                except queue.Empty:
                                    return None
                            
                            chunk_data = await asyncio.to_thread(get_chunk)
                            
                            if chunk_data is None:
                                # Queue empty, check if stream is done
                                if stream_done and chunk_queue.empty():
                                    # Stream finished but no sentinel received - might be an error
                                    if stream_error:
                                        yield f"data: {json.dumps({'type': 'error', 'error': stream_error})}\n\n"
                                    break
                                # Yield control to event loop and check again
                                await asyncio.sleep(0.01)
                                continue
                            elif isinstance(chunk_data, tuple) and chunk_data[0] == "error":
                                # Error occurred
                                yield f"data: {json.dumps({'type': 'error', 'error': chunk_data[1]})}\n\n"
                                break
                            else:
                                # Valid content chunk
                                content = chunk_data
                                full_response += content
                                # Stream each token as it arrives
                                try:
                                    yield f"data: {json.dumps({'type': 'token', 'job_index': idx, 'content': content})}\n\n"
                                except Exception as yield_error:
                                    logger.error(f"Error yielding token: {yield_error}", exc_info=True)
                                    break
                        except Exception as e:
                            logger.error(f"Error in streaming loop: {e}", exc_info=True)
                            break
                    # Ensure stream task completes
                    try:
                        await stream_task
                    except Exception:
                        pass  # Task may have already completed or failed
                except Exception as e:
                    logger.error(f"Error in stream processing: {e}", exc_info=True)
                    try:
                        await stream_task
                    except Exception:
                        pass
                
                # Parse the response
                from agents import extract_json_from_response
                data_result = extract_json_from_response(full_response)
                
                if not data_result or data_result.get("match_score") is None:
                    data_result = data_result or {}
                    data_result["match_score"] = 0.5
                
                score = float(data_result.get("match_score", 0.5))
                requirements_satisfied_list = data_result.get("requirements_satisfied", []) or []
                requirements_missing_list = data_result.get("requirements_missing", []) or []
                requirements_met = int(data_result.get("requirements_met", 0))
                total_requirements = int(data_result.get("total_requirements", 0))
                
                if total_requirements == 0:
                    total_requirements = len(requirements_satisfied_list) + len(requirements_missing_list)
                    if total_requirements == 0:
                        total_requirements = 1
                    requirements_met = len(requirements_satisfied_list)
                
                if requirements_met > total_requirements:
                    requirements_met = total_requirements
                
                scored_jobs.append({
                    "job": job,
                    "match_score": score,
                    "key_matches": data_result.get("key_matches", []) or [],
                    "requirements_met": requirements_met,
                    "total_requirements": total_requirements,
                    "requirements_satisfied": requirements_satisfied_list,
                    "requirements_missing": requirements_missing_list,
                    "improvements_needed": data_result.get("improvements_needed", []) or [],
                    "reasoning": data_result.get("reasoning", "Score calculated based on candidate-job alignment"),
                })
                
                yield f"data: {json.dumps({'type': 'job_complete', 'job_index': idx, 'match_score': score, 'job_title': job.job_title})}\n\n"
            
            # Sort by score
            scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
            top_matches = [s for s in scored_jobs if s["match_score"] >= 0.5][:10]
            
            if not top_matches:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No jobs matched with score >= 0.5'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'status': 'summarizing', 'message': f'Summarizing {len(top_matches)} matched job(s)...'})}\n\n"
            
            # Create matched jobs (simplified summarization)
            matched_jobs_list = []
            for idx, entry in enumerate(top_matches, 1):
                job = entry["job"]
                score = entry["match_score"]
                
                # Create summary
                summary_parts = [f"Match score: {score:.1%}."]
                if entry.get("reasoning"):
                    summary_parts.append(entry["reasoning"][:200])
                if entry.get("key_matches"):
                    matches_str = ', '.join(entry["key_matches"][:3])
                    summary_parts.append(f"Key matches: {matches_str}.")
                
                summary_text = " ".join(summary_parts)
                if len(summary_text) > 500:
                    summary_text = summary_text[:497] + "..."
                
                matched_job = {
                    "rank": idx,
                    "job_url": str(job.url),
                    "job_title": job.job_title or "Unknown",
                    "company": job.company or "Unknown",
                    "match_score": round(score, 3),
                    "summary": summary_text,
                    "key_matches": entry["key_matches"],
                    "requirements_met": entry["requirements_met"],
                    "total_requirements": entry["total_requirements"],
                    "requirements_satisfied": entry["requirements_satisfied"],
                    "requirements_missing": entry["requirements_missing"],
                    "improvements_needed": entry["improvements_needed"],
                    "location": None,
                    "scraped_summary": None
                }
                matched_jobs_list.append(matched_job)
            
            # Check sponsorship (simplified)
            sponsorship_info = None
            if jobs and jobs[0].company:
                from sponsorship_checker import check_sponsorship
                cleaned_name = clean_company_name(jobs[0].company)
                if cleaned_name:
                    try:
                        sponsorship_result = await asyncio.to_thread(
                            check_sponsorship, cleaned_name, jobs[0].description, openai_key
                        )
                        if sponsorship_result:
                            sponsorship_info = {
                                "company_name": sponsorship_result.get("company_name"),
                                "sponsors_workers": sponsorship_result.get("sponsors_workers", False),
                                "visa_types": sponsorship_result.get("visa_types"),
                                "summary": sponsorship_result.get("summary", "No sponsorship information available")
                            }
                    except:
                        pass
            
            # Send final response
            processing_time = f"{time.time() - start_time:.1f}s"
            final_response = {
                "candidate_profile": candidate_profile,
                "matched_jobs": matched_jobs_list,
                "processing_time": processing_time,
                "jobs_analyzed": len(jobs),
                "request_id": request_id,
                "sponsorship": sponsorship_info
            }
            
            yield f"data: {json.dumps({'type': 'complete', 'response': final_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            logger.error(f"Streaming error: {error_msg}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg, 'traceback': traceback.format_exc()})}\n\n"
    
    return StreamingResponse(
        generate_stream(resume_bytes_from_file),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/")

async def root():

    return {"status": "ok", "version": "0.1.0"}





# Firebase Resume Endpoints

@app.post("/api/firebase/resumes", response_model=FirebaseResumeListResponse)

async def get_user_resumes(request: GetUserResumesRequest):

    """

    Fetch all resumes for a specific user from Firebase Firestore.

    

    Request Body:

        user_id: The user ID to fetch resumes for

        

    Returns:

        List of resumes for the user

    """

    try:

        from firebase_service import get_firebase_service

        

        firebase_service = get_firebase_service()

        resumes_data = firebase_service.get_user_resumes(request.user_id)

        

        # Convert to Pydantic models with better error handling
        resumes = []
        for idx, resume_data in enumerate(resumes_data):
            try:
                # Ensure 'id' field is present
                if "id" not in resume_data:
                    logger.warning(f"Resume at index {idx} missing 'id' field, skipping")
                    continue
                
                # Convert to FirebaseResume model
                resume = FirebaseResume(**resume_data)
                resumes.append(resume)
            except ValidationError as ve:
                logger.error(f"Validation error for resume at index {idx}: {ve}")
                logger.error(f"Resume data: {resume_data}")
                # Skip invalid resumes but continue processing others
                continue
            except Exception as e:
                logger.error(f"Error processing resume at index {idx}: {e}")
                logger.error(f"Resume data: {resume_data}")
                # Skip problematic resumes but continue processing others
                continue

        

        return FirebaseResumeListResponse(

            user_id=request.user_id,

            resumes=resumes,

            count=len(resumes)

        )

    except ImportError as e:

        logger.error(f"Import error in get_user_resumes: {e}")
        raise HTTPException(

            status_code=500,

            detail=f"Firebase service not available: {str(e)}"

        )

    except ValidationError as ve:
        logger.error(f"Validation error in get_user_resumes: {ve}")
        raise HTTPException(

            status_code=422,

            detail=f"Invalid request data: {str(ve)}"

        )

    except Exception as e:

        logger.error(f"Error in get_user_resumes: {e}", exc_info=True)
        raise HTTPException(

            status_code=500,

            detail=f"Failed to fetch resumes: {str(e)}"

        )





@app.post("/api/firebase/resumes/get", response_model=FirebaseResumeResponse)

async def get_user_resume(request: GetUserResumeRequest):

    """

    Fetch a specific resume by ID for a user from Firebase Firestore.

    

    Request Body:

        user_id: The user ID

        resume_id: The resume document ID

        

    Returns:

        The resume document

    """

    try:

        from firebase_service import get_firebase_service

        

        firebase_service = get_firebase_service()

        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)

        

        if not resume_data:

            raise HTTPException(

                status_code=404,

                detail=f"Resume {request.resume_id} not found for user {request.user_id}"

            )

        

        return FirebaseResumeResponse(

            user_id=request.user_id,

            resume=FirebaseResume(**resume_data)

        )

    except HTTPException:

        raise

    except ImportError as e:

        raise HTTPException(

            status_code=500,

            detail=f"Firebase service not available: {str(e)}"

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Failed to fetch resume: {str(e)}"

        )





@app.post("/api/firebase/resumes/pdf")

async def get_user_resume_pdf(request: GetUserResumePdfRequest):

    """

    Fetch a resume PDF as raw bytes (decoded from base64).

    

    Request Body:

        user_id: The user ID

        resume_id: The resume document ID

        

    Returns:

        PDF file as bytes with appropriate content-type

    """

    try:

        from fastapi.responses import Response

        from firebase_service import get_firebase_service

        

        firebase_service = get_firebase_service()

        pdf_bytes = firebase_service.get_resume_pdf_bytes(request.user_id, request.resume_id)

        

        if not pdf_bytes:

            raise HTTPException(

                status_code=404,

                detail=f"Resume PDF not found for user {request.user_id}, resume {request.resume_id}"

            )

        

        return Response(

            content=pdf_bytes,

            media_type="application/pdf",

            headers={

                "Content-Disposition": f'attachment; filename="resume_{request.resume_id}.pdf"'

            }

        )

    except HTTPException:

        raise

    except ImportError as e:

        raise HTTPException(

            status_code=500,

            detail=f"Firebase service not available: {str(e)}"

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Failed to fetch resume PDF: {str(e)}"

        )





@app.post("/api/firebase/resumes/base64")

async def get_user_resume_base64(request: GetUserResumeBase64Request):

    """

    Fetch a resume PDF as base64 string (with PDF_BASE64: prefix removed).

    

    Request Body:

        user_id: The user ID

        resume_id: The resume document ID

        

    Returns:

        JSON with base64 string

    """

    try:

        from firebase_service import get_firebase_service

        

        firebase_service = get_firebase_service()

        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)

        

        if not resume_data:

            raise HTTPException(

                status_code=404,

                detail=f"Resume {request.resume_id} not found for user {request.user_id}"

            )

        

        base64_content = firebase_service.extract_pdf_base64(resume_data)

        

        if not base64_content:

            raise HTTPException(

                status_code=404,

                detail=f"Resume PDF content not found for user {request.user_id}, resume {request.resume_id}"

            )

        

        return {

            "user_id": request.user_id,

            "resume_id": request.resume_id,

            "base64": base64_content

        }

    except HTTPException:

        raise

    except ImportError as e:

        raise HTTPException(

            status_code=500,

            detail=f"Firebase service not available: {str(e)}"

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Failed to fetch resume base64: {str(e)}"

        )





@app.post("/api/firebase/users/saved-cvs", response_model=SavedCVResponse)

async def get_user_saved_cvs(request: GetUserSavedCvsRequest):

    """

    Fetch the savedCVs array for a user from Firebase Firestore.

    

    This endpoint retrieves the savedCVs array stored at the user document level.

    

    Request Body:

        user_id: The user ID

        

    Returns:

        The savedCVs array for the user

    """

    try:

        from firebase_service import get_firebase_service

        

        firebase_service = get_firebase_service()

        saved_cvs = firebase_service.get_user_saved_cvs(request.user_id)

        

        return SavedCVResponse(

            user_id=request.user_id,

            saved_cvs=saved_cvs,

            count=len(saved_cvs)

        )

    except ImportError as e:

        raise HTTPException(

            status_code=500,

            detail=f"Firebase service not available: {str(e)}"

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Failed to fetch savedCVs: {str(e)}"

        )





# Job Information Extraction Endpoint

@app.post("/api/extract-job-info", response_model=JobInfoExtracted)

async def extract_job_info(

    request: ExtractJobInfoRequest,

    settings: Settings = Depends(get_settings)

):

    """

    Extract job title, company name, portal, and description from a job posting URL.

    Uses enhanced HTML parsing with multiple extraction methods including JSON-LD,

    portal-specific selectors, AI fallback, and agent-based description generation.

    Also surfaces visa or scholarship information when mentioned.
    """

    try:

        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")

        os.environ.setdefault("FIRECRAWL_API_KEY", settings.firecrawl_api_key or "")

        

        job_info = extract_job_info_from_url(str(request.job_url), settings.firecrawl_api_key)

        visa_info = job_info.get("visa_scholarship_info") or "Not specified"
        

        description = None

        if settings.openai_api_key:

            try:

                print(f"[AGENT] Generating job description for: {request.job_url}")

                from agents import build_scraper, build_summarizer


                scraper_agent = build_scraper()
                scrape_prompt = (
                    f"Extract all job posting details from this URL: {request.job_url}\n\n"
                    "Provide complete job description, requirements, responsibilities, and any other relevant information."
                )
                scrape_response = scraper_agent.run(scrape_prompt)

                

                scraped_content = ""

                if hasattr(scrape_response, "content"):
                    scraped_content = str(scrape_response.content)

                elif hasattr(scrape_response, "messages") and scrape_response.messages:
                    last_msg = scrape_response.messages[-1]

                    scraped_content = str(last_msg.content if hasattr(last_msg, "content") else last_msg)
                else:

                    scraped_content = str(scrape_response)

                

                print(f"[AGENT] [SCRAPER] Scraped {len(scraped_content)} characters")

                

                if scraped_content:

                    summarizer_agent = build_summarizer(settings.model_name)

                    summary_prompt = (
                        "Create a concise, professional job description summary (150-250 words) from this scraped job posting content.\n\n"
                        f"Job Title: {job_info.get('job_title', 'Not specified')}\n"
                        f"Company: {job_info.get('company_name', 'Not specified')}\n\n"
                        "Scraped Content:\n"
                        f"{scraped_content[:4000]}\n\n"
                        "Generate a clear, well-structured summary that includes:\n"
                        "- Key responsibilities\n"
                        "- Required qualifications and skills\n"
                        "- Preferred experience level\n"
                        "- Any notable benefits or details\n\n"
                        "Keep it professional and informative, suitable for displaying to job seekers."
                    )
                    summary_response = summarizer_agent.run(summary_prompt)

                    

                    if hasattr(summary_response, "content"):
                        description = str(summary_response.content).strip()

                    elif hasattr(summary_response, "messages") and summary_response.messages:
                        last_msg = summary_response.messages[-1]

                        description = str(last_msg.content if hasattr(last_msg, "content") else last_msg).strip()
                    else:

                        description = str(summary_response).strip()

                    

                    # Clean markdown formatting inconsistencies
                    description = clean_summary_text(description)
                    

                    print(f"[AGENT] [SUMMARIZER] Generated description ({len(description)} characters)")


                    visa_keywords = [
                        "visa sponsorship",
                        "visa support",
                        "scholarship",
                        "h1b",
                        "work permit",
                        "financial support",
                        "tuition assistance",
                        "visa assistance",
                    ]
                    desc_lower = description.lower()
                    if any(kw in desc_lower for kw in visa_keywords):
                        for keyword in visa_keywords:
                            if keyword in desc_lower:
                                idx = desc_lower.find(keyword)
                                start = max(0, idx - 100)
                                end = min(len(description), idx + len(keyword) + 200)
                                visa_info = description[start:end].strip()
                                break
                else:

                    print("[AGENT] [WARNING] No scraped content received from scraper agent")
                    

            except Exception as agent_error:

                print(f"[AGENT] [ERROR] Failed to generate description (non-fatal): {agent_error}")

                import traceback


                print(f"[AGENT] Traceback: {traceback.format_exc()}")

        

        if description:

            job_info["description"] = description
        

        if not job_info.get("job_title") or len(job_info.get("job_title", "")) < 3:
            if settings.openai_api_key:

                try:

                    if not requests or not BeautifulSoup:

                        return JobInfoExtracted(**job_info)

                    

                    headers = {

                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept-Language": "en-US,en;q=0.9",
                    }

                    resp = requests.get(str(request.job_url), headers=headers, timeout=20)

                    if resp.ok:

                        soup = BeautifulSoup(resp.text, "lxml")
                        main_content = ""
                        main_elem = soup.find("main") or soup.find("article") or soup.find("body")
                        if main_elem:

                            main_content = main_elem.get_text(strip=True)[:3000]
                        

                        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")

                        from agents import build_resume_parser

                        

                        extractor_agent = build_resume_parser(settings.model_name)

                        prompt = (
                            "Extract ONLY the job title from this job posting page content. Return ONLY the job title text, nothing else.\n\n"
                            f"Page content:\n{main_content[:2000]}\n\n"
                            "Return ONLY the job title (e.g., \"Software Engineer\", \"Data Scientist\", \"Product Manager\"), no explanations, no quotes, no markdown."
                        )
                        ai_response = extractor_agent.run(prompt)

                        

                        if hasattr(ai_response, "content"):
                            ai_title = str(ai_response.content).strip()

                        elif hasattr(ai_response, "messages") and ai_response.messages:
                            last_msg = ai_response.messages[-1]

                            ai_title = str(last_msg.content if hasattr(last_msg, "content") else last_msg).strip()
                        else:

                            ai_title = str(ai_response).strip()

                        

                        ai_title = ai_title.strip('"\'')
                        ai_title = re.sub(r"^.*title[:\s]*", "", ai_title, flags=re.I)

                        if ai_title and 3 <= len(ai_title) <= 100:

                            if not any(bad in ai_title.lower() for bad in ["i cannot", "i don't", "unable to", "sorry", "error"]):
                                job_info["job_title"] = ai_title
                                job_info["success"] = True

                        if main_content and (not visa_info or visa_info.lower() == "not specified"):
                            visa_lower = main_content.lower()
                            for keyword in [
                                "visa sponsorship",
                                "visa support",
                                "scholarship",
                                "h1b",
                                "work permit",
                                "financial support",
                                "tuition assistance",
                            ]:
                                if keyword in visa_lower:
                                    idx = visa_lower.find(keyword)
                                    start = max(0, idx - 100)
                                    end = min(len(main_content), idx + len(keyword) + 200)
                                    visa_info = main_content[start:end].strip()
                                    break

                except Exception as ai_error:
                    print(f"AI fallback failed (non-fatal): {ai_error}")

        # Note: visa_scholarship_info is kept internally for sponsorship checking but not returned in response
        job_info.setdefault("success", True)
        job_info.setdefault("error", None)

        return JobInfoExtracted(**job_info)
        

    except Exception as e:

        return JobInfoExtracted(

            job_url=str(request.job_url),

            job_title=None,

            company_name=None,

            portal=detect_portal(str(request.job_url)),

            visa_scholarship_info="Not specified",

            success=False,

            error=str(e),
        )


# Apollo API People Search Endpoint

@app.post("/api/apollo/search-people", response_model=ApolloPersonSearchResponse)
async def apollo_search_people(
    request: ApolloPersonSearchRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Search for people using Apollo API.
    
    Requires APOLLO_API_KEY environment variable or api_key in request.
    Returns people with email and phone number access based on your Apollo plan.
    """
    try:
        # Get API key from request or environment
        api_key = request.api_key or os.getenv("APOLLO_API_KEY")
        
        if not api_key:
            return ApolloPersonSearchResponse(
                success=False,
                error="Apollo API key not provided. Set APOLLO_API_KEY environment variable or provide api_key in request.",
                people=[],
            )
        
        # Build request payload
        payload = {}
        
        if request.person_titles:
            payload["person_titles"] = request.person_titles
        if request.person_locations:
            payload["person_locations"] = request.person_locations
        if request.organization_names:
            payload["organization_names"] = request.organization_names
        if request.person_emails:
            payload["person_emails"] = request.person_emails
        if request.person_names:
            payload["person_names"] = request.person_names
        if request.page:
            payload["page"] = request.page
        if request.per_page:
            payload["per_page"] = request.per_page
        
        # Apollo API endpoint
        url = "https://api.apollo.io/api/v1/mixed_people/search"
        
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Api-Key": api_key,  # Apollo requires API key in header for security
        }
        
        # Note: Do NOT include api_key in payload - Apollo requires it in X-Api-Key header only
        
        # Make API request
        print(f"[Apollo API] Searching for people with filters: {list(payload.keys())}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if not response.ok:
            error_msg = f"Apollo API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", error_data.get("message", error_msg))
                
                # Check if it's a plan limitation error
                if "free plan" in error_msg.lower() or "upgrade" in error_msg.lower():
                    error_msg += " Note: The search endpoint requires a paid plan. Use /api/apollo/enrich-person for free plan enrichment."
            except:
                error_msg = f"{error_msg} - {response.text[:200]}"
            
            return ApolloPersonSearchResponse(
                success=False,
                error=error_msg,
                people=[],
            )
        
        # Parse response
        data = response.json()
        
        # Extract people data
        people_list = []
        for person_data in data.get("people", []):
            person = {
                "id": person_data.get("id", ""),
                "first_name": person_data.get("first_name"),
                "last_name_obfuscated": person_data.get("last_name_obfuscated"),
                "title": person_data.get("title"),
                "last_refreshed_at": person_data.get("last_refreshed_at"),
                "has_email": person_data.get("has_email"),
                "has_city": person_data.get("has_city"),
                "has_state": person_data.get("has_state"),
                "has_country": person_data.get("has_country"),
                "has_direct_phone": person_data.get("has_direct_phone"),
                "email": person_data.get("email"),  # May be None if not accessible
                "phone_number": person_data.get("phone_numbers", [{}])[0].get("raw_number") if person_data.get("phone_numbers") else None,
            }
            
            # Add organization if present
            if person_data.get("organization"):
                person["organization"] = {
                    "name": person_data.get("organization", {}).get("name"),
                    "has_industry": person_data.get("organization", {}).get("has_industry"),
                    "has_phone": person_data.get("organization", {}).get("has_phone"),
                    "has_city": person_data.get("organization", {}).get("has_city"),
                    "has_state": person_data.get("organization", {}).get("has_state"),
                    "has_country": person_data.get("organization", {}).get("has_country"),
                    "has_zip_code": person_data.get("organization", {}).get("has_zip_code"),
                    "has_revenue": person_data.get("organization", {}).get("has_revenue"),
                    "has_employee_count": person_data.get("organization", {}).get("has_employee_count"),
                }
            
            people_list.append(person)
        
        print(f"[Apollo API] Found {len(people_list)} people (total: {data.get('total_entries', 0)})")
        
        return ApolloPersonSearchResponse(
            total_entries=data.get("total_entries"),
            people=people_list,
            page=request.page or 1,
            per_page=request.per_page or 25,
            success=True,
        )
        
    except requests.exceptions.RequestException as e:
        return ApolloPersonSearchResponse(
            success=False,
            error=f"Network error: {str(e)}",
            people=[],
        )
    except Exception as e:
        print(f"[Apollo API] Error: {e}")
        import traceback
        traceback.print_exc()
        return ApolloPersonSearchResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            people=[],
        )


# Apollo API Person Enrichment Endpoint (Works on Free Plan)

@app.post("/api/apollo/enrich-person", response_model=ApolloEnrichPersonResponse)
async def apollo_enrich_person(
    request: ApolloEnrichPersonRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Enrich person data using Apollo API (works on free plan).
    
    Requires at least one of: email, first_name+last_name, or domain.
    Returns enriched person data with email, phone, and organization information.
    
    Free Plan Limits:
    - 50 calls per minute
    - 200 calls per hour
    - 600 calls per day
    """
    try:
        # Get API key from request or environment
        api_key = request.api_key or os.getenv("APOLLO_API_KEY")
        
        if not api_key:
            return ApolloEnrichPersonResponse(
                success=False,
                error="Apollo API key not provided. Set APOLLO_API_KEY environment variable or provide api_key in request.",
            )
        
        # Validate that at least one identifier is provided
        if not request.email and not (request.first_name and request.last_name) and not request.domain:
            return ApolloEnrichPersonResponse(
                success=False,
                error="At least one identifier required: email, (first_name + last_name), or domain",
            )
        
        # Build request payload
        payload = {}
        
        if request.email:
            payload["email"] = request.email
        if request.first_name:
            payload["first_name"] = request.first_name
        if request.last_name:
            payload["last_name"] = request.last_name
        if request.domain:
            payload["domain"] = request.domain
        
        # Apollo API endpoint for enrichment (works on free plan)
        url = "https://api.apollo.io/api/v1/people/match"
        
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
        }
        
        # Make API request
        print(f"[Apollo API] Enriching person with: {list(payload.keys())}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if not response.ok:
            error_msg = f"Apollo API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", error_data.get("message", error_msg))
            except:
                error_msg = f"{error_msg} - {response.text[:200]}"
            
            return ApolloEnrichPersonResponse(
                success=False,
                error=error_msg,
            )
        
        # Parse response
        data = response.json()
        
        # Extract person data
        person_data = data.get("person", {})
        
        if not person_data:
            return ApolloEnrichPersonResponse(
                success=False,
                error="No person data found in response",
            )
        
        person = {
            "id": person_data.get("id", ""),
            "first_name": person_data.get("first_name"),
            "last_name_obfuscated": person_data.get("last_name_obfuscated"),
            "title": person_data.get("title"),
            "last_refreshed_at": person_data.get("last_refreshed_at"),
            "has_email": person_data.get("has_email"),
            "has_city": person_data.get("has_city"),
            "has_state": person_data.get("has_state"),
            "has_country": person_data.get("has_country"),
            "has_direct_phone": person_data.get("has_direct_phone"),
            "email": person_data.get("email"),  # May be None if not accessible
            "phone_number": person_data.get("phone_numbers", [{}])[0].get("raw_number") if person_data.get("phone_numbers") else None,
        }
        
        # Add organization if present
        if person_data.get("organization"):
            person["organization"] = {
                "name": person_data.get("organization", {}).get("name"),
                "has_industry": person_data.get("organization", {}).get("has_industry"),
                "has_phone": person_data.get("organization", {}).get("has_phone"),
                "has_city": person_data.get("organization", {}).get("has_city"),
                "has_state": person_data.get("organization", {}).get("has_state"),
                "has_country": person_data.get("organization", {}).get("has_country"),
                "has_zip_code": person_data.get("organization", {}).get("has_zip_code"),
                "has_revenue": person_data.get("organization", {}).get("has_revenue"),
                "has_employee_count": person_data.get("organization", {}).get("has_employee_count"),
            }
        
        print(f"[Apollo API] Successfully enriched person: {person.get('first_name')} {person.get('last_name_obfuscated')}")
        
        return ApolloEnrichPersonResponse(
            person=person,
            success=True,
        )
        
    except requests.exceptions.RequestException as e:
        return ApolloEnrichPersonResponse(
            success=False,
            error=f"Network error: {str(e)}",
        )
    except Exception as e:
        print(f"[Apollo API] Error: {e}")
        import traceback
        traceback.print_exc()
        return ApolloEnrichPersonResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
        )


# Sponsorship Check Endpoint

@app.post("/api/check-sponsorship", response_model=SponsorshipInfo)
async def check_sponsorship_endpoint(
    request: SponsorshipCheckRequest,
    settings: Settings = Depends(get_settings),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Check if a company sponsors workers for UK visas.
    
    This endpoint uses the EXACT SAME process as the match-jobs endpoint:
    1. Receives job_info (scraped job data)
    2. Pre-extracts company name using multiple strategies
    3. Uses LLM agent (summarize_scraped_data) to extract structured info including company_name
    4. Checks UK visa sponsorship database using fuzzy matching
    5. Uses AI agent to select correct company match
    6. Optionally fetches additional company info from web
    7. Builds enhanced summary combining CSV and web data
    
    Args:
        request: SponsorshipCheckRequest with job_info (scraped job data)
        settings: Application settings
    
    Returns:
        SponsorshipInfo with sponsorship details (same format as match-jobs endpoint)
    """
    try:
        logger.info("Checking company sponsorship status")
        
        # Get job_info (scraped job data) - same as match-jobs endpoint
        job_data = request.job_info
        
        if not job_data:
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="job_info field is required and cannot be empty.",
            )
        
        # STEP 1: Extract company name using OpenAI (same as match-jobs)
        openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="OpenAI API key is required for company name extraction.",
            )
        
        logger.debug(f"Extracting company name from {len(job_data)} chars via OpenAI")
        
        # Use the same extraction function as match-jobs
        from job_extractor import extract_company_and_title_from_raw_data
        
        try:
            extracted_info = await asyncio.to_thread(
                extract_company_and_title_from_raw_data,
                job_data,  # Send raw scraped data as-is, no preprocessing
                openai_key,
                "gpt-4o-mini"  # Use fast and intelligent OpenAI model
            )
            
            # Get extracted company name
            extracted_company = extracted_info.get("company_name")
            logger.debug(f"OpenAI extracted company: {extracted_company[:50] if extracted_company else 'None'}")
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}, trying fallback methods")
            extracted_company = None
        
        # STEP 2: Clean and validate company name (same as match-jobs)
        final_company = None
        if extracted_company:
            cleaned = clean_company_name(extracted_company)
            if cleaned and len(cleaned) >= 2 and cleaned.lower() not in ["not specified", "unknown", "none"]:
                final_company = cleaned
                logger.debug(f"Using OpenAI-extracted company: {final_company}")
        
        # Fallback: Try sponsorship_checker extract_company_name if OpenAI failed
        if not final_company:
            try:
                from sponsorship_checker import extract_company_name
                extracted = extract_company_name(job_data[:2000] if job_data else "")
                if extracted:
                    cleaned = clean_company_name(extracted)
                    if cleaned and len(cleaned) >= 2:
                        final_company = cleaned
                        logger.debug(f"Using fallback extracted company: {final_company}")
            except Exception as e:
                logger.debug(f"Fallback extraction error: {e}")
        
        if not final_company or final_company == "Company name not available in posting":
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="Company name could not be extracted from the provided job_info. The LLM agent was unable to identify a company name in the job posting data.",
            )
        
        # STEP 3: Check sponsorship using cached CSV data (same as match-jobs)
        from sponsorship_checker import check_sponsorship, get_company_info_from_web
        
        logger.info(f"Checking sponsorship for company: {final_company}")
        
        # Use async thread pool for check_sponsorship (same as match-jobs)
        sponsorship_result = await asyncio.to_thread(
            check_sponsorship,
            final_company,
            job_data,
            openai_key
        )
        
        # STEP 4: Get company info from web (same as match-jobs)
        company_info_summary = None
        matched_company_name = sponsorship_result.get('company_name') or final_company
        if matched_company_name and matched_company_name.lower() not in ["unknown", "not specified", "none", ""]:
            try:
                if openai_key:
                    logger.debug(f"Fetching additional company information from web for {matched_company_name}")
                    company_info_summary = await asyncio.to_thread(
                        get_company_info_from_web,
                        matched_company_name,
                        openai_key
                    )
                else:
                    logger.debug("OpenAI API key not available, skipping web search")
            except Exception as e:
                logger.debug(f"Error fetching company info from web: {e}")
                # Continue without web info - not critical
        
        # Build enhanced summary (same as match-jobs)
        base_summary = sponsorship_result.get('summary', 'No sponsorship information available')
        # Clean and normalize the base summary
        base_summary = clean_summary_text(base_summary)
        enhanced_summary = base_summary
        
        if company_info_summary:
            # Clean company info
            company_info_cleaned = clean_summary_text(company_info_summary)
            
            # Remove redundant visa sponsorship information from company info
            # (since we already have it confirmed from CSV)
            sponsors_workers = sponsorship_result.get('sponsors_workers', False)
            if sponsors_workers:
                # Split into sentences and filter out redundant ones about visa sponsorship
                sentences = re.split(r'[.!?]+', company_info_cleaned)
                filtered_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence or len(sentence) < 10:
                        continue
                    
                    sentence_lower = sentence.lower()
                    
                    # Skip sentences about visa sponsorship that are uncertain/redundant
                    # since we already have confirmed info from CSV
                    if any(phrase in sentence_lower for phrase in ['visa sponsorship', 'visa sponsor', 'visa not', 'visa information']):
                        # If it mentions uncertainty or suggests contacting, skip it
                        if any(uncertain_phrase in sentence_lower for uncertain_phrase in [
                            'not found', 'was not found', 'not available', 'uncertain',
                            'potentially', 'generally', 'might', 'may', 'could', 'contact',
                            'check', 'definitive information', 'advisable', 'check their',
                            'hr department', 'official careers', 'cannot be filled'
                        ]):
                            continue  # Skip this redundant sentence
                    
                    filtered_sentences.append(sentence)
                
                # Rejoin sentences
                company_info_cleaned = '. '.join(filtered_sentences)
                if company_info_cleaned and not company_info_cleaned.endswith(('.', '!', '?')):
                    company_info_cleaned += '.'
                company_info_cleaned = re.sub(r'\s+', ' ', company_info_cleaned).strip()
            
            # Only append company info if there's substantial unique content
            # (avoid repeating what's already in base_summary)
            if company_info_cleaned and len(company_info_cleaned.strip()) > 30:
                # Check for overlap with base_summary to avoid duplication
                if base_summary:
                    base_lower = base_summary.lower()
                    company_lower = company_info_cleaned.lower()
                    
                    # Simple overlap check - if too similar, skip
                    # Count common significant words (longer than 4 chars)
                    base_words = {w for w in base_lower.split() if len(w) > 4}
                    company_words = {w for w in company_lower.split() if len(w) > 4}
                    common_words = base_words & company_words
                    
                    # If more than 40% overlap in significant content, don't duplicate
                    if len(common_words) > 0 and len(common_words) / max(len(company_words), 1) > 0.4:
                        # Just use base summary to avoid repetition
                        enhanced_summary = base_summary
                    else:
                        # Add unique company information
                        enhanced_summary = f"{base_summary}. {company_info_cleaned}"
                else:
                    enhanced_summary = company_info_cleaned
            else:
                # Not enough content, just use base summary
                enhanced_summary = base_summary
            
            # Normalize whitespace and remove duplicate periods
            enhanced_summary = re.sub(r'\s+', ' ', enhanced_summary)
            enhanced_summary = re.sub(r'\.\s*\.', '.', enhanced_summary)  # Remove double periods
            enhanced_summary = enhanced_summary.strip()
            logger.debug("Enhanced summary with company information from web (removed redundant visa sponsorship info)")
        
        # Return SponsorshipInfo (same format as match-jobs)
        return SponsorshipInfo(
            company_name=sponsorship_result.get('company_name'),
            sponsors_workers=sponsorship_result.get('sponsors_workers', False),
            visa_types=sponsorship_result.get('visa_types'),
            summary=enhanced_summary
        )
        
    except FileNotFoundError as e:
        logger.error(f"Sponsorship database not available: {e}")
        return SponsorshipInfo(
            company_name=None,
            sponsors_workers=False,
            visa_types=None,
            summary=f"Sponsorship database not available: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error checking sponsorship: {e}", exc_info=True)
        return SponsorshipInfo(
            company_name=None,
            sponsors_workers=False,
            visa_types=None,
            summary=f"Error checking sponsorship: {str(e)}",
        )


# Playwright Scraper Endpoint with Agent Summarization

@app.post("/api/playwright-scrape", response_model=PlaywrightScrapeResponse)

async def playwright_scrape(

    json_body: Optional[str] = Form(default=None),

    pdf_file: Optional[UploadFile] = File(default=None),

    settings: Settings = Depends(get_settings)

):

    """

    Scrape a job posting URL using Playwright, summarize with an agent, and score against resume.

    

    This endpoint:

    1. Accepts form data: pdf_file (resume, optional) and json_body (with url and optional user_id)

    2. Uses Playwright to scrape the job posting page

    3. Extracts structured data (title, company, description, etc.)

    4. Uses an agent to summarize and structure the information

    5. Parses the resume and scores the job-candidate match

    6. Returns scraped data, summarized data, and match score

    

    Request Form Data:

        pdf_file: Resume PDF file (optional, required for scoring)

        json_body: JSON string with {"url": "https://...", "user_id": "..."}; user_id is optional

        

    Returns:

        - url: The scraped URL

        - scraped_data: Raw scraped data from Playwright

        - summarized_data: Structured data from agent summarization

        - match_score: Job-candidate match score (0.0-1.0) if resume provided, otherwise null

        - key_matches: Key matching qualifications (null when scoring is skipped)

        - requirements_met: Number of requirements met (null when scoring is skipped)

        - total_requirements: Total number of requirements (null when scoring is skipped)

        - reasoning: Reasoning for the match score (null when scoring is skipped)

        - success: Whether scraping and summarization was successful

        - error: Error message if any

    """

    portal: Optional[str] = None
    authorized_sponsor: Optional[bool] = None
    try:

        from playwright.sync_api import sync_playwright

        from scrapers.response import summarize_scraped_data

        

        # STEP 1: Parse request data

        if not json_body:

            raise HTTPException(status_code=400, detail="Missing json_body field")

        

        # Parse JSON body

        try:

            clean_json = json_body.strip().strip('"').replace('\\"', '"')

            payload = json.loads(clean_json)

            url = payload.get("url")

            if not url:

                raise HTTPException(status_code=400, detail="Missing 'url' in json_body")

        except json.JSONDecodeError as e:

            raise HTTPException(status_code=400, detail=f"Invalid JSON in json_body: {e}")

        

        # STEP 2: Handle optional resume and user information

        user_id = payload.get("user_id")

        user_id_provided = bool(user_id)

        if user_id_provided:

            print(f"[INFO] Received user_id: {user_id}")

        

        resume_provided = pdf_file is not None

        scoring_enabled = False

        candidate_profile: Optional[CandidateProfile] = None

        

        if not resume_provided and not user_id_provided:

            print("[INFO] No resume PDF or user_id provided; running in scrape-and-summarize mode only.")

        elif resume_provided:

            resume_bytes = await pdf_file.read()

            resume_text = extract_text_from_pdf_bytes(resume_bytes)

            

            if not resume_text or len(resume_text.strip()) < 50:

                raise HTTPException(status_code=400, detail="Resume PDF is empty or could not be extracted")

            

            print(f"\n{'='*80}")

            print(f"RESUME PARSER - Processing resume ({len(resume_text)} chars)")

            print(f"{'='*80}")

            

            # Parse resume using agent

            parser_agent = build_resume_parser(settings.model_name)

            resume_prompt = f"Parse this resume and extract structured information:\n\n{resume_text}"

            resume_response = parser_agent.run(resume_prompt)

            

            # Extract resume JSON

            if hasattr(resume_response, 'content'):

                response_text = str(resume_response.content)

            elif hasattr(resume_response, 'messages') and resume_response.messages:

                last_msg = resume_response.messages[-1]

                response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)

            else:

                response_text = str(resume_response)

            

            response_text = response_text.strip()

            resume_json = extract_json_from_response(response_text)

            

            if not resume_json:

                raise HTTPException(status_code=500, detail="Failed to parse resume")

            

            # Create candidate profile

            exp_summary = resume_json.get("experience_summary")

            if isinstance(exp_summary, (list, dict)):

                exp_summary = json.dumps(exp_summary, indent=2)

            elif exp_summary is None:

                exp_summary = "Not provided"

            

            total_years = parse_experience_years(resume_json.get("total_years_experience"))

            

            candidate_profile = CandidateProfile(

                name=resume_json.get("name") or "Unknown",

                email=resume_json.get("email"),

                phone=resume_json.get("phone"),

                skills=resume_json.get("skills", []) or [],

                experience_summary=exp_summary,

                total_years_experience=total_years,

                interests=resume_json.get("interests", []) or [],

                education=resume_json.get("education", []) or [],

                certifications=resume_json.get("certifications", []) or [],

                raw_text_excerpt=redact_long_text(resume_text, 300),

            )

            scoring_enabled = True

        else:

            # user_id provided without resume

            print("[INFO] No resume uploaded; skipping resume parsing and match scoring.")

        

        print(f"\n{'='*80}")

        print(f"PLAYWRIGHT SCRAPER - Scraping: {url}")

        print(f"{'='*80}")

        

        # STEP 3: Scrape with Playwright

        def scrape_with_playwright(url: str) -> Dict[str, Any]:

            """Synchronous Playwright scraping function with enhanced extraction."""
            import re

            

            with sync_playwright() as p:

                browser = p.chromium.launch(headless=True)

                page = browser.new_page()

                
                # Set a realistic viewport and user agent
                page.set_viewport_size({"width": 1920, "height": 1080})
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })
                
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                except:
                    # If initial load fails, try with networkidle
                    try:
                        page.goto(url, wait_until="networkidle", timeout=60000)
                    except:
                        pass
                
                # Wait for page to load with multiple strategies
                try:
                    page.wait_for_load_state("networkidle", timeout=30000)

                except:
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=30000)
                    except:
                        pass
                
                # Additional wait for dynamic content (especially for LinkedIn)
                import time
                time.sleep(3)  # Give JavaScript time to render
                
                # Try to wait for specific elements that indicate the page is loaded
                try:
                    # Wait for either job title or description to appear
                    page.wait_for_selector('h1, .jobs-description, .job-description, [data-test-id*="description"]', timeout=10000)
                except:
                    pass  # Continue even if selectors don't appear
                

                # Get page title

                page_title = page.title()

                current_url = page.url

                detected_portal = detect_portal(current_url)
                html_content = page.content()

                
                # Get text content - try multiple methods
                text_content = ""
                try:
                    text_content = page.inner_text("body")

                except:
                    try:
                        # Fallback: get text from main content area
                        main_content = page.query_selector("main") or page.query_selector("#main") or page.query_selector("body")
                        if main_content:
                            text_content = main_content.inner_text()
                    except:
                        # Last resort: extract from HTML
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_content, 'lxml')
                        text_content = soup.get_text(separator='\n', strip=True)
                
                # Extract job title with more selectors and strategies
                # Includes selectors from content script for LinkedIn, Indeed, Internshala, etc.
                job_title = None

                title_selectors = [

                    # LinkedIn selectors
                    '.job-details-jobs-unified-top-card__job-title',
                    '.jobs-unified-top-card__job-title',
                    '.jobs-details-top-card__job-title',

                    'h1[data-test-id*="job-title"]',

                    '.topcard__title',

                    'h1.jobs-details-top-card__job-title',
                    'h1.job-title',
                    # Internshala selectors
                    '.heading_2_4',
                    '.heading_4_5.profile',
                    # Indeed selectors
                    '.jobsearch-JobInfoHeader-title',
                    # Civil Service Jobs
                    '#id_common_page_title_h1',
                    '.csr-page-title h1',
                    # Reed
                    '[data-qa="job-title"]',
                    '.job-title-block_title__9fRYc',
                    # StudentJob
                    'h1[itemprop="title"]',
                    'p[itemprop="title"]',
                    '.h4[itemprop="title"]',
                    '.job-opening__title h1',
                    # JustEngineers/Jobsite/TotalJobs
                    '[data-at="header-job-title"]',
                    # JobsACUK
                    '.j-advert__title',
                    # Generic fallbacks
                    'h1',

                    '.job-title',
                    '[data-test-id="job-title"]',
                    'h2.job-title'
                ]

                for selector in title_selectors:

                    try:

                        element = page.query_selector(selector)

                        if element:

                            job_title = element.inner_text().strip()

                            if job_title and len(job_title) > 3 and len(job_title) < 200:
                                # Validate it's not a search page title
                                if not re.search(r'\d+[,\d]*\+?\s*jobs?\s+in', job_title, re.I):
                                    break

                    except:

                        continue

                

                # Extract company name with content script selectors
                company_name = None

                company_selectors = [

                    # LinkedIn selectors
                    '.jobs-details-top-card__company-name',

                    '.jobs-details-top-card__company-name-link',
                    '.jobs-details-top-card__company-info',
                    '[data-test-id*="company"]',

                    '[data-test-id="job-poster"]',
                    '.topcard__org-name',

                    # Generic
                    '.company-name',

                    'a[href*="/company/"]',

                    # JobsACUK
                    '.j-advert__employer',
                    # JSON-LD fallback (will be checked later)
                ]

                for selector in company_selectors:

                    try:

                        element = page.query_selector(selector)

                        if element:

                            company_name = element.inner_text().strip()

                            if company_name and len(company_name) > 2 and len(company_name) < 100:
                                break

                    except:

                        continue

                

                # Try to extract from JSON-LD if not found
                if not company_name:
                    try:
                        scripts = page.query_selector_all('script[type="application/ld+json"]')
                        for script in scripts:
                            try:
                                data = json.loads(script.inner_text())
                                if data.get('hiringOrganization', {}).get('name'):
                                    company_name = data['hiringOrganization']['name']
                                    break
                            except:
                                continue
                    except:
                        pass
                
                if not company_name and job_title:

                    company_match = re.search(rf'{re.escape(job_title)}\n([A-Za-z0-9\s&]+)\s+([A-Za-z,\s]+,\s*[A-Za-z,\s]+)', text_content)

                    if company_match:

                        company_name = company_match.group(1).strip()

                

                # Extract job description with enhanced selectors from content script
                description = None

                desc_selectors = [

                    # LinkedIn selectors (from content script)
                    '.jobs-search__job-details--wrapper',
                    '.job-view-layout',
                    '.jobs-details',
                    '.jobs-details__main-content',
                    '#job-details',
                    '.jobs-description__container',
                    '.jobs-description-content',
                    '.jobs-description__text',

                    '.jobs-box__html-content',

                    # Internshala selectors
                    '.individual_internship_header',
                    '.individual_internship_details',
                    '.tags_container_outer',
                    '.applications_message_container',
                    '.internship_details',
                    '.activity_section',
                    '.detail_view',
                    # Indeed selectors
                    '#jobDescriptionText',
                    '.jobsearch-JobComponent-description',
                    '#jobsearch-ViewjobPaneWrapper',
                    '.jobsearch-embeddedBody',
                    '.jobsearch-BodyContainer',
                    '.jobsearch-JobComponent',
                    '.fastviewjob',
                    # Civil Service Jobs
                    '.vac_display_panel_main_inner',
                    '.vac_display_panel_side_inner',
                    '#main-content',
                    # Reed
                    '[data-qa="job-details-drawer-modal-body"]',
                    # StudentJob
                    '[data-job-openings-sticky-title-target="jobOpeningContent"]',
                    '.job-opening__body',
                    '.job-opening__description',
                    '.printable',
                    '.card__body',
                    '.sticky-title__moving-target',
                    # JustEngineers/Jobsite/TotalJobs
                    '[data-at="job-ad-header"]',
                    '[data-at="job-ad-content"]',
                    '.at-section-text-jobDescription',
                    '.job-ad-display-ofzx2',
                    '.job-ad-display-cl9qsc',
                    '.job-ad-display-kyg8or',
                    '.job-ad-display-nfizss',
                    '.listingContentBrandingColor',
                    '.job-ad-display-1b1is8w',
                    # Milkround
                    '[data-at="content-container"]',
                    "[data-at='section-text-jobDescription']",
                    "[data-at='section-text-jobDescription-content']",
                    '.job-ad-display-n10qeq',
                    '.job-ad-display-tt0ywc',
                    '.job-ad-display-gro348',
                    # WorkInStartups
                    'main.container',
                    '.ui-adp-content',
                    '.ui-job-card-info',
                    '.adp-body',
                    # CharityJob
                    '.job-details-summary',
                    '.job-description-wrapper',
                    '.job-description',
                    '.job-organisation-profile',
                    '.job-attachments',
                    '.job-post-summary',
                    '.job-detail-foot-note',
                    # JobsACUK
                    '.j-advert-details__container',
                    '#job-description',
                    # Generic fallbacks
                    '[data-test-id*="description"]',

                    '.job-description',

                    '#job-description',

                    '.jobs-description-content__text',

                    'div[data-test-id*="job-details"]',
                    '.jobs-description-content',
                    '[id*="job-details"]',
                    '[class*="job-description"]',
                    '[class*="description"]',
                    'section[data-test-id*="description"]',
                    'div.jobs-description'
                ]
                
                for selector in desc_selectors:

                    try:

                        element = page.query_selector(selector)

                        if element:

                            description = element.inner_text().strip()

                            if description and len(description) > 100:  # Ensure we got substantial content
                                break

                    except:

                        continue

                

                # If still no description, try getting HTML and extracting from multiple elements
                # Use content script approach: collect HTML from multiple wrapper elements
                if not description or len(description) < 100:
                    try:
                        # Content script approach: collect HTML from multiple wrappers and merge
                        wrapper_selectors = [
                            # LinkedIn wrappers (from content script)
                            '.jobs-search__job-details--wrapper',
                            '.job-view-layout',
                            '.jobs-details',
                            '.jobs-details__main-content',
                            # Internshala wrappers
                            '.individual_internship_header',
                            '.individual_internship_details',
                            '.detail_view',
                            # Indeed wrappers
                            '#jobDescriptionText',
                            '.jobsearch-JobComponent-description',
                            '#jobsearch-ViewjobPaneWrapper',
                            # Civil Service Jobs
                            '.vac_display_panel_main_inner',
                            '.vac_display_panel_side_inner',
                            # Reed
                            '[data-qa="job-details-drawer-modal-body"]',
                            # StudentJob
                            '[data-job-openings-sticky-title-target="jobOpeningContent"]',
                            '.job-opening__body',
                            # JustEngineers/Jobsite/TotalJobs
                            '[data-at="job-ad-content"]',
                            # WorkInStartups
                            'main.container',
                            # CharityJob
                            '.job-description-wrapper',
                            # JobsACUK
                            '.j-advert-details__container'
                        ]
                        
                        html_parts = []
                        seen_elements = set()
                        
                        for wrapper_sel in wrapper_selectors:
                            try:
                                elements = page.query_selector_all(wrapper_sel)
                                for elem in elements:
                                    # Use element's unique identifier to avoid duplicates
                                    elem_id = id(elem)
                                    if elem_id not in seen_elements:
                                        seen_elements.add(elem_id)
                                        try:
                                            # Get innerHTML using evaluate
                                            html = elem.evaluate('el => el.innerHTML')
                                            if html and html.strip():
                                                html_parts.append(html)
                                        except:
                                            # Fallback to inner_text if inner_html fails
                                            try:
                                                text = elem.inner_text().strip()
                                                if text and len(text) > 100:
                                                    html_parts.append(text)
                                            except:
                                                continue
                            except:
                                continue
                        
                        # If we collected HTML parts, merge and extract text
                        if html_parts:
                            merged_html = '\n\n'.join(html_parts)
                            # Convert HTML to text (similar to content script)
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(merged_html, 'lxml')
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            # Get text
                            merged_text = soup.get_text(separator='\n', strip=True)
                            # Clean up (similar to content script clean function)
                            merged_text = re.sub(r'\n{3,}', '\n\n', merged_text)
                            merged_text = re.sub(r'[ \t]{2,}', ' ', merged_text)
                            merged_text = merged_text.replace('\u00A0', ' ').strip()
                            
                            if len(merged_text) > 100:
                                description = merged_text
                        
                        # Fallback: query all description-related elements
                        if not description or len(description) < 100:
                            desc_elements = page.query_selector_all('[class*="description"], [id*="description"], [data-test-id*="description"]')
                            for elem in desc_elements:
                                try:
                                    text = elem.inner_text().strip()
                                    if text and len(text) > 100:
                                        description = text
                                        break
                                except:
                                    continue
                    except Exception as e:
                        print(f"[DEBUG] Error in wrapper extraction: {e}")
                        pass
                
                # Fallback: Extract from text content using regex
                if not description or len(description) < 100:
                    # Look for "Job Description" or "About the job" sections
                    desc_patterns = [
                        r'(?:Job Description|About the job|Description)\s*\n\n(.*?)(?:\n\n(?:Additional Information|Show more|Qualifications|Requirements|Similar jobs|Referrals)|\Z)',
                        r'(?:Job Description|About the job|Description)\s*\n\n(.*)',
                    ]
                    for pattern in desc_patterns:
                        desc_match = re.search(pattern, text_content, re.DOTALL | re.IGNORECASE)
                        if desc_match:
                            description = desc_match.group(1).strip()
                            if len(description) > 100:
                                break
                
                # Last resort: Extract from main content area
                if not description or len(description) < 100:
                    try:
                        # Get main content area
                        main = page.query_selector("main") or page.query_selector("#main") or page.query_selector("body")
                        if main:
                            main_text = main.inner_text()
                            # Try to extract meaningful content (skip navigation, headers, etc.)
                            lines = main_text.split('\n')
                            desc_lines = []
                            in_description = False
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                # Skip short lines that are likely navigation
                                if len(line) < 20 and not in_description:
                                    continue
                                # Start collecting when we see description-like content
                                if any(keyword in line.lower() for keyword in ['description', 'about', 'role', 'responsibilities', 'requirements']):
                                    in_description = True
                                if in_description:
                                    desc_lines.append(line)
                                    if len('\n'.join(desc_lines)) > 500:  # Got enough content
                                        break
                            if desc_lines:
                                description = '\n'.join(desc_lines)
                    except:
                        pass
                

                # Extract location

                location = None

                location_selectors = [

                    '.jobs-details-top-card__primary-description-without-tagline',

                    '.jobs-details-top-card__bullet',

                    '[data-test-id*="location"]',
                    '.job-criteria__text',
                    '[data-test-id="job-location"]'
                ]

                for selector in location_selectors:

                    try:

                        element = page.query_selector(selector)

                        if element:

                            location = element.inner_text().strip()

                            if location:
                                break

                    except:

                        continue

                

                if not location and company_name:

                    location_match = re.search(rf'{re.escape(company_name)}\s+([A-Za-z,\s]+,\s*[A-Za-z,\s]+)', text_content)

                    if location_match:

                        location = location_match.group(1).strip()

                

                # Extract qualifications and skills

                qualifications = None

                skills = None

                

                if text_content:

                    qual_match = re.search(r'Qualifications\s*\n\n(.*?)(?:\n\nSuggested skills|\n\nAdditional Information|\Z)', text_content, re.DOTALL)

                    if qual_match:

                        qualifications = qual_match.group(1).strip()

                    

                    skills_match = re.search(r'Suggested skills\s*\n\n(.*?)(?:\n\nAdditional Information|\Z)', text_content, re.DOTALL)

                    if skills_match:

                        skills = skills_match.group(1).strip()

                

                # Extract structured data from JSON-LD (like content script)
                json_ld_data = ""
                try:
                    scripts = page.query_selector_all('script[type="application/ld+json"]')
                    structured_lines = []
                    
                    for script in scripts:
                        try:
                            script_text = script.inner_text()
                            if not script_text:
                                continue
                            data = json.loads(script_text)
                            
                            # Extract job posting data
                            if data.get("@type") == "JobPosting" or "JobPosting" in str(data.get("@type", [])):
                                if data.get("title") and not job_title:
                                    job_title = data["title"]
                                if data.get("hiringOrganization", {}).get("name") and not company_name:
                                    company_name = data["hiringOrganization"]["name"]
                                if data.get("jobLocation") and not location:
                                    loc = data["jobLocation"]
                                    if isinstance(loc, list):
                                        loc = loc[0] if loc else {}
                                    if isinstance(loc, dict):
                                        addr = loc.get("address", {})
                                        if isinstance(addr, dict):
                                            loc_parts = [
                                                addr.get("addressLocality"),
                                                addr.get("addressRegion"),
                                                addr.get("postalCode"),
                                                addr.get("addressCountry")
                                            ]
                                            location = ", ".join([p for p in loc_parts if p])
                                
                                # Build structured data string
                                if data.get("title"):
                                    structured_lines.append(f"Title: {data['title']}")
                                if data.get("hiringOrganization", {}).get("name"):
                                    structured_lines.append(f"Company: {data['hiringOrganization']['name']}")
                                if data.get("datePosted"):
                                    structured_lines.append(f"Posted: {data['datePosted']}")
                                if data.get("validThrough"):
                                    structured_lines.append(f"Apply By: {data['validThrough']}")
                                if data.get("employmentType"):
                                    emp_type = data["employmentType"]
                                    if isinstance(emp_type, list):
                                        emp_type = ", ".join(emp_type)
                                    structured_lines.append(f"Employment: {emp_type}")
                                if data.get("baseSalary"):
                                    salary = data["baseSalary"]
                                    if isinstance(salary, dict) and salary.get("value"):
                                        val = salary["value"]
                                        if isinstance(val, dict):
                                            currency = salary.get("currency", "")
                                            min_val = val.get("minValue") or val.get("value")
                                            max_val = val.get("maxValue")
                                            unit = val.get("unitText", "")
                                            if max_val:
                                                structured_lines.append(f"Salary: {currency} {min_val} - {max_val} / {unit}")
                                            else:
                                                structured_lines.append(f"Salary: {currency} {min_val} / {unit}")
                                if data.get("skills"):
                                    skills_data = data["skills"]
                                    if isinstance(skills_data, list):
                                        skills_str = ", ".join(skills_data)
                                    else:
                                        skills_str = str(skills_data)
                                    structured_lines.append(f"Skills: {skills_str}")
                                if data.get("industry"):
                                    structured_lines.append(f"Industry: {data['industry']}")
                                if data.get("totalJobOpenings"):
                                    structured_lines.append(f"Openings: {data['totalJobOpenings']}")
                            
                            # Also add raw JSON for reference
                            json_ld_data += "\n" + json.dumps(data, indent=2)
                            
                        except:
                            continue
                    
                    # Add structured data to description if we have it
                    if structured_lines:
                        structured_text = "\n\n— Structured Data (JSON-LD) —\n" + "\n".join(structured_lines)
                        if description:
                            description = description + structured_text
                        else:
                            description = structured_text
                    
                except Exception as json_error:
                    print(f"[DEBUG] Error extracting JSON-LD: {json_error}")
                
                # Final description: merge description with JSON-LD raw if available
                final_description = description or text_content[:5000]
                if json_ld_data.strip() and len(json_ld_data) < 5000:
                    final_description = final_description + "\n\n--- JSON-LD Raw ---\n" + json_ld_data
                
                scraped_data = {

                    "url": current_url,

                    "title": page_title,

                    "job_title": job_title,

                    "company_name": company_name,

                    "location": location,

                    "description": final_description,
                    "qualifications": qualifications,

                    "suggested_skills": skills,

                    "text_content": text_content,

                    "html_length": len(html_content),
                    "portal": detected_portal,
                }

                

                browser.close()

            

            return scraped_data

        

        # STEP 3: Normalize URL (extract actual job URL from search URLs)
        actual_url = url
        if "linkedin.com/jobs/search" in url.lower() and "currentJobId" in url:
            # Extract job ID from LinkedIn search URL
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            job_id = params.get("currentJobId", [None])[0]
            if job_id:
                actual_url = f"https://www.linkedin.com/jobs/view/{job_id}"
                print(f"[INFO] Detected LinkedIn search URL, converting to job posting URL: {actual_url}")
        
        # STEP 4: Scrape with Playwright (with fallbacks)
        scraped_data = None
        scraping_method = "playwright"
        scraping_error = None
        
        try:
            scraped_data = await asyncio.to_thread(scrape_with_playwright, actual_url)
            text_content = scraped_data.get("text_content") or scraped_data.get("description") or ""
            job_title = scraped_data.get("job_title") or scraped_data.get("title") or ""
            
            # Validation: Check if we got a valid job posting (not a search page or error page)
            is_valid_job = True
            validation_errors = []
            
            # Check 1: Sufficient content (minimum 500 characters)
            if len(text_content) < 500:
                validation_errors.append(f"Insufficient content ({len(text_content)} chars)")
                is_valid_job = False
            
            # Check 2: Valid job title (not search page indicators)
            invalid_title_patterns = [
                r'\d+[,\d]*\+?\s*jobs?\s+in',  # "3,874,000+ Jobs in United States"
                r'search\s+results?',
                r'find\s+jobs?',
                r'job\s+search',
                r'jobs?\s+on\s+linkedin',
            ]
            if job_title:
                title_lower = job_title.lower()
                for pattern in invalid_title_patterns:
                    if re.search(pattern, title_lower, re.I):
                        validation_errors.append(f"Invalid job title (search page detected): {job_title}")
                        is_valid_job = False
                        break
            
            # Check 3: Blocking pages
            block_indicators = ["request blocked", "you have been blocked", "cloudflare", "access denied", "please verify"]
            text_lower = text_content.lower()
            if any(indicator in text_lower for indicator in block_indicators):
                validation_errors.append("Page appears to be blocked")
                is_valid_job = False
            
            # Check 4: Description should not be too short
            description = scraped_data.get("description") or ""
            if len(description) < 100:
                validation_errors.append(f"Description too short ({len(description)} chars)")
                is_valid_job = False
            
            if not is_valid_job:
                print(f"[WARNING] Playwright validation failed: {', '.join(validation_errors)}")
                print(f"[WARNING] Trying Firecrawl fallback...")
                scraping_error = f"Playwright validation failed: {', '.join(validation_errors)}"
                scraped_data = None
            else:
                print(f"[SUCCESS] Playwright validation passed")
        except Exception as e:
            print(f"[ERROR] Playwright scraping failed: {e}")
            scraping_error = str(e)
            scraped_data = None
        
        # FALLBACK 1: Try Firecrawl if Playwright failed or insufficient content
        if scraped_data is None:
            print(f"\n{'='*80}")
            print(f"FIRECRAWL FALLBACK - Attempting to scrape with Firecrawl")
            print(f"{'='*80}")
            scraping_method = "firecrawl"
            
            try:
                firecrawl_api_key = settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
                if firecrawl_api_key:
                    def scrape_with_firecrawl(scrape_url: str) -> Dict[str, Any]:
                        """Scrape using Firecrawl."""
                        fc_result = scrape_website_custom(scrape_url, firecrawl_api_key)
                        
                        if isinstance(fc_result, dict) and 'error' not in fc_result:
                            content = str(fc_result.get('content') or fc_result.get('markdown') or fc_result.get('text') or "")
                            html_content = fc_result.get('html') or ""
                            metadata = fc_result.get('metadata') or {}
                            
                            # Extract title and company from metadata
                            title = metadata.get('title') or ""
                            if not title and html_content:
                                try:
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(html_content, 'lxml')
                                    if soup.title:
                                        title = soup.title.string.strip() if soup.title.string else ""
                                except:
                                    pass
                            
                            # Parse HTML for better extraction if available
                            job_title = None
                            company_name = None
                            description = None
                            
                            if html_content:
                                try:
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(html_content, 'lxml')
                                    
                                    # Extract job title
                                    title_selectors = [
                                        'h1.job-title', 'h2.job-title', '.job-title',
                                        '[data-testid*="job-title"]', 'h1', 'h2'
                                    ]
                                    for selector in title_selectors:
                                        elem = soup.select_one(selector)
                                        if elem and elem.get_text(strip=True):
                                            job_title = elem.get_text(strip=True)
                                            break
                                    
                                    # Extract company name
                                    company_selectors = [
                                        '.company-name', '[class*="Company"]',
                                        '[data-testid*="company"]', 'a[href*="/company/"]'
                                    ]
                                    for selector in company_selectors:
                                        elem = soup.select_one(selector)
                                        if elem and elem.get_text(strip=True):
                                            company_name = elem.get_text(strip=True)
                                            break
                                    
                                    # Extract description
                                    desc_selectors = [
                                        '.job-description', '#job-description',
                                        '[data-testid*="description"]', '.jobs-description'
                                    ]
                                    for selector in desc_selectors:
                                        elem = soup.select_one(selector)
                                        if elem and elem.get_text(strip=True):
                                            description = elem.get_text(strip=True)
                                            break
                                except Exception as parse_error:
                                    print(f"[Firecrawl] HTML parsing error: {parse_error}")
                            
                            # Use content as description if not found
                            if not description and content:
                                description = content[:5000]  # Limit description length
                            
                            return {
                                "url": scrape_url,
                                "title": title,
                                "job_title": job_title or title,
                                "company_name": company_name,
                                "location": None,
                                "description": description or content[:2000],
                                "qualifications": None,
                                "suggested_skills": None,
                                "text_content": content,
                                "html_length": len(html_content),
                                "portal": detect_portal(scrape_url),
                            }
                        else:
                            raise Exception(fc_result.get('error', 'Unknown Firecrawl error'))
                    
                    scraped_data = await asyncio.to_thread(scrape_with_firecrawl, actual_url)
                    text_content = scraped_data.get("text_content") or scraped_data.get("description") or ""
                    
                    # Check if Firecrawl got enough content
                    if len(text_content) < 500:
                        print(f"[WARNING] Firecrawl returned insufficient content ({len(text_content)} chars), trying DuckDuckGo fallback...")
                        scraping_error = f"Insufficient content from Firecrawl ({len(text_content)} chars)"
                        scraped_data = None
                    else:
                        print(f"[SUCCESS] Firecrawl scraped {len(text_content)} characters")
                else:
                    print(f"[WARNING] Firecrawl API key not available, skipping Firecrawl fallback")
            except Exception as e:
                print(f"[ERROR] Firecrawl scraping failed: {e}")
                scraping_error = str(e)
                scraped_data = None
        
        # FALLBACK 2: Try DuckDuckGo web search if both Playwright and Firecrawl failed
        if scraped_data is None:
            print(f"\n{'='*80}")
            print(f"DUCKDUCKGO FALLBACK - Attempting web search")
            print(f"{'='*80}")
            scraping_method = "duckduckgo"
            
            try:
                openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                if openai_key:
                    def search_with_duckduckgo(search_url: str) -> Dict[str, Any]:
                        """Search for job information using DuckDuckGo."""
                        from agents import Agent, get_model_config
                        from langchain_community.tools import DuckDuckGoSearchRun
                        
                        # Create search agent
                        model = get_model_config("gpt-4o-mini", default_temperature=0)
                        search_agent = Agent(
                            name="Job Search Agent",
                            model=model,
                            tools=[DuckDuckGoSearchRun()],
                            instructions=[
                                "Search for information about this job posting URL.",
                                "Extract: job title, company name, job description, requirements, and location.",
                                "Provide comprehensive information about the job posting.",
                                "Keep the response detailed and informative."
                            ],
                            show_tool_calls=False,
                            markdown=False,
                        )
                        
                        # Search query
                        query = f"Find detailed information about this job posting: {search_url}. Extract job title, company name, description, requirements, and location."
                        
                        # Get response
                        response = search_agent.run(query, stream=False)
                        
                        # Extract content
                        search_content = None
                        if hasattr(response, 'content'):
                            search_content = str(response.content)
                        elif isinstance(response, str):
                            search_content = response
                        else:
                            search_content = str(response)
                        
                        # Try to extract structured info from the response
                        job_title = None
                        company_name = None
                        description = search_content
                        
                        # Simple extraction patterns
                        title_match = re.search(r'(?:Job Title|Title|Position)[:\s]+([^\n]+)', search_content, re.I)
                        if title_match:
                            job_title = title_match.group(1).strip()
                        
                        company_match = re.search(r'(?:Company|Employer|Organization)[:\s]+([^\n]+)', search_content, re.I)
                        if company_match:
                            company_name = company_match.group(1).strip()
                        
                        # Extract description section
                        desc_match = re.search(r'(?:Description|Job Description|Details)[:\s]+(.*?)(?:\n\n|\n[A-Z][a-z]+:|$)', search_content, re.I | re.DOTALL)
                        if desc_match:
                            description = desc_match.group(1).strip()
                        
                        return {
                            "url": search_url,
                            "title": job_title or "Job Posting",
                            "job_title": job_title,
                            "company_name": company_name,
                            "location": None,
                            "description": description or search_content[:2000],
                            "qualifications": None,
                            "suggested_skills": None,
                            "text_content": search_content,
                            "html_length": 0,
                            "portal": detect_portal(search_url),
                        }
                    
                    scraped_data = await asyncio.to_thread(search_with_duckduckgo, actual_url)
                    text_content = scraped_data.get("text_content") or scraped_data.get("description") or ""
                    
                    if len(text_content) < 200:
                        print(f"[WARNING] DuckDuckGo returned insufficient content ({len(text_content)} chars)")
                        scraping_error = f"All scraping methods failed. Last attempt (DuckDuckGo) returned only {len(text_content)} chars"
                    else:
                        print(f"[SUCCESS] DuckDuckGo search returned {len(text_content)} characters")
                else:
                    print(f"[WARNING] OpenAI API key not available, skipping DuckDuckGo fallback")
                    scraping_error = "All scraping methods failed. OpenAI API key required for DuckDuckGo fallback"
            except Exception as e:
                print(f"[ERROR] DuckDuckGo search failed: {e}")
                scraping_error = f"All scraping methods failed. Last error: {str(e)}"
                scraped_data = None
        
        # If all methods failed, return error response
        if scraped_data is None:
            error_message = scraping_error or "All scraping methods failed"
            print(f"[ERROR] {error_message}")
            return PlaywrightScrapeResponse(
                url=url,
                scraped_data={},
                summarized_data={},
                portal=detect_portal(url),
                is_authorized_sponsor=None,
                match_score=None,
                key_matches=None,
                requirements_met=None,
                total_requirements=None,
                reasoning=None,
                visa_scholarship_info="Not specified",
                success=False,
                error=error_message,
            )
        
        portal = scraped_data.get("portal") or detect_portal(url)
        authorized_sponsor = is_authorized_sponsor(scraped_data.get("company_name"))
        
        print(f"[INFO] Successfully scraped using {scraping_method.upper()} method")
        print(f"[INFO] Content length: {len(scraped_data.get('text_content') or scraped_data.get('description') or '')} characters")
        

        print(f"\n{'='*80}")

        print(f"AGENT SUMMARIZATION - Processing scraped data")

        print(f"{'='*80}")

        

        # STEP 4: Summarize scraped data

        openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if not openai_key:

            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        

        summarized_data = await asyncio.to_thread(summarize_scraped_data, scraped_data, openai_key)

        if isinstance(summarized_data, dict):
            summarized_data.setdefault("portal", portal)
        
        # STEP 5: Create JobPosting from summarized data with proper cleaning
        # Extract and clean job title
        summarized_title = summarized_data.get("job_title")
        scraped_title = scraped_data.get("job_title")
        text_content = scraped_data.get("text_content", "") or scraped_data.get("description", "")
        final_job_title = extract_job_title_from_content(text_content, summarized_title or scraped_title)
        if not final_job_title or final_job_title == "Job title not available in posting":
            final_job_title = clean_job_title(summarized_title) or clean_job_title(scraped_title) or "Job title not available in posting"
        
        # Extract and clean company name
        summarized_company = summarized_data.get("company_name")
        scraped_company = scraped_data.get("company_name")
        final_company = extract_company_name_from_content(text_content, summarized_company or scraped_company)
        if not final_company or final_company == "Company name not available in posting":
            final_company = clean_company_name(summarized_company) or clean_company_name(scraped_company) or "Company name not available in posting"
        
        job = JobPosting(

            url=url,

            job_title=final_job_title,
            company=final_company,
            description=summarized_data.get("description") or scraped_data.get("description") or scraped_data.get("text_content", "")[:2000],

            skills_needed=summarized_data.get("required_skills", []) or [],

            experience_level=summarized_data.get("required_experience"),

            salary=summarized_data.get("salary")

        )

        authorized_sponsor = is_authorized_sponsor(job.company)
        if isinstance(summarized_data, dict):
            summarized_data.setdefault("is_authorized_sponsor", authorized_sponsor)
        if isinstance(scraped_data, dict):
            scraped_data.setdefault("is_authorized_sponsor", authorized_sponsor)
        

        scoring_result: Optional[Dict[str, Any]] = None

        

        if scoring_enabled and candidate_profile:

            print(f"\n{'='*80}")

            print(f"JOB SCORER - Calculating match score")

            print(f"{'='*80}")



            # STEP 6: Score the job

            scorer_agent = build_scorer(settings.model_name)



            def score_job_sync() -> Optional[Dict[str, Any]]:

                """Score the job using AI reasoning."""

                try:

                    prompt = f"""

Analyze the match between candidate and job. Consider ALL requirements from the job description.



Candidate Profile:

{json.dumps(candidate_profile.dict(), indent=2)}



Job Details:

- Title: {job.job_title}

- Company: {job.company}

- URL: {str(job.url)}

- Description: {job.description[:2000]}



CRITICAL: Read the job description carefully. If this is a:

- Billing/Finance role: Score based on financial/accounting skills

- Tech/Engineering role: Score based on technical skills

- Sales/Marketing role: Score based on communication/business skills



Return ONLY valid JSON (no markdown) with:

{{

  "match_score": 0.75,

  "key_matches": ["skill1", "skill2"],

  "requirements_met": 5,

  "total_requirements": 8,

  "reasoning": "Brief explanation of score"

}}



Be strict with scoring:

- < 0.3: Poor fit (major skill gaps)

- 0.3-0.5: Weak fit (some alignment)

- 0.5-0.7: Good fit (strong alignment)

- > 0.7: Excellent fit (ideal candidate)

"""

                    response = scorer_agent.run(prompt)



                    if hasattr(response, 'content'):

                        response_text = str(response.content)

                    elif hasattr(response, 'messages') and response.messages:

                        last_msg = response.messages[-1]

                        response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)

                    else:

                        response_text = str(response)



                    response_text = response_text.strip()

                    data = extract_json_from_response(response_text)



                    if not data or data.get("match_score") is None:

                        data = data or {}

                        data["match_score"] = 0.5



                    score = float(data.get("match_score", 0.5))

                    print(f"[OK] Match Score: {score:.1%}")



                    return {

                        "match_score": score,

                        "key_matches": data.get("key_matches", []) or [],

                        "requirements_met": int(data.get("requirements_met", 0)),

                        "total_requirements": int(data.get("total_requirements", 1)),

                        "reasoning": data.get("reasoning", "Score calculated based on candidate-job alignment"),

                    }

                except Exception as e:

                    print(f"[ERROR] Error scoring job: {e}")

                    import traceback

                    traceback.print_exc()

                    return None



            scoring_result = await asyncio.to_thread(score_job_sync)



            if not scoring_result:

                scoring_result = {

                    "match_score": 0.5,

                    "key_matches": [],

                    "requirements_met": 0,

                    "total_requirements": 1,

                    "reasoning": "Unable to calculate match score",

                }

        else:

            if scoring_enabled and not candidate_profile:

                print("[WARNING] Scoring was enabled but candidate profile could not be created.")

            else:

                print("[INFO] Skipping match scoring because resume data is not available.")



        # STEP 6B: Persist to Firebase when resume and user_id are provided

        firebase_doc_id: Optional[str] = None

        if scoring_enabled and user_id_provided and scoring_result:

            try:

                # Reload environment variables to ensure Firebase credentials are accessible

                load_dotenv()

                load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)



                from firebase_service import get_firebase_service



                firebase_service = get_firebase_service()



                match_score = scoring_result.get("match_score")

                key_matches = scoring_result.get("key_matches", []) or []

                reasoning = scoring_result.get("reasoning", "")

                summary_description = summarized_data.get("description") or scraped_data.get("description") or ""



                job_description_parts: List[str] = []

                if match_score is not None:

                    job_description_parts.append(f"Match Score: {match_score:.1%}")



                requirements_met = scoring_result.get("requirements_met")

                total_requirements = scoring_result.get("total_requirements")

                if requirements_met is not None and total_requirements:

                    try:

                        req_percentage = (requirements_met / total_requirements) * 100

                        job_description_parts.append(

                            f"Requirements Met: {requirements_met}/{total_requirements} ({req_percentage:.0f}%)"

                        )

                    except ZeroDivisionError:

                        job_description_parts.append(

                            f"Requirements Met: {requirements_met}/{total_requirements}"

                        )



                if summary_description:

                    desc_text = summary_description

                    if len(desc_text) > 1000:

                        desc_text = desc_text[:1000] + "..."

                    job_description_parts.append(desc_text)



                if key_matches:

                    job_description_parts.append("Key Matches: " + ", ".join(key_matches[:10]))



                if reasoning:

                    job_description_parts.append(f"Scoring Reasoning: {reasoning}")



                job_description = "\n\n".join(job_description_parts)

                notes = summary_description[:500] if summary_description else reasoning[:500]



                visa_info = summarized_data.get("visa_scholarship_info") or "Not specified"

                visa_required = "Yes" if visa_info and visa_info.lower() not in {"not specified", "no"} else "No"



                job_data = {

                    "appliedDate": datetime.now(),

                    "company": job.company or "",

                    "createdAt": datetime.now(),

                    "interviewDate": "",

                    "jobDescription": job_description,

                    "link": str(job.url),

                    "notes": notes,

                    "portal": portal,

                    "role": job.job_title or "",

                    "status": "Matched",

                    "visaRequired": visa_required,

                    "authorizedSponsor": authorized_sponsor,
                }



                print(f"\n{'='*80}")

                print("[SAVE] Persisting Playwright match to Firestore")

                print(f"User ID: {user_id}")

                print(f"Job Title: {job.job_title}")

                print(f"Company: {job.company}")

                print(f"Portal: {portal}")

                print(f"{'='*80}")



                firebase_doc_id = firebase_service.save_job_application(user_id, job_data)



                print(f"[SUCCESS] Saved job application with document ID: {firebase_doc_id}")



            except ImportError as import_error:

                print(f"[WARNING] Firebase service not available: {import_error}")

                print("[INFO] Install firebase-admin and ensure credentials are configured.")

            except Exception as save_error:

                print(f"[ERROR] Failed to save Playwright job application: {save_error}")

                import traceback

                print(traceback.format_exc())



        print(f"\n{'='*80}")

        if scoring_result:

            print("SUCCESS - Scraping, summarization, and scoring completed")

        else:

            print("SUCCESS - Scraping and summarization completed")

        print(f"{'='*80}")

        # Note: visa_scholarship_info is kept internally for sponsorship checking but not returned in response
        

        return PlaywrightScrapeResponse(

            url=url,

            scraped_data=scraped_data,

            summarized_data=summarized_data,

            portal=portal,
            is_authorized_sponsor=authorized_sponsor,
            match_score=scoring_result["match_score"] if scoring_result else None,

            key_matches=scoring_result["key_matches"] if scoring_result else None,

            requirements_met=scoring_result["requirements_met"] if scoring_result else None,

            total_requirements=scoring_result["total_requirements"] if scoring_result else None,

            reasoning=scoring_result["reasoning"] if scoring_result else None,

            success=True,

            error=None

        )

        

    except HTTPException:

        raise

    except Exception as e:

        import traceback

        error_msg = str(e)

        traceback.print_exc()

        print(f"\n{'='*80}")

        print(f"ERROR - {error_msg}")

        print(f"{'='*80}")

        

        return PlaywrightScrapeResponse(

            url=url if 'url' in locals() else "",

            scraped_data={},

            summarized_data={},

            portal=portal,
            is_authorized_sponsor=authorized_sponsor,
            match_score=None,

            key_matches=None,

            requirements_met=None,

            total_requirements=None,

            reasoning=None,

            success=False,

            error=error_msg

        )





# Summarizer-Only Endpoint

@app.post("/api/summarize-job", response_model=SummarizeJobResponse)

async def summarize_job(

    request: SummarizeJobRequest,

    settings: Settings = Depends(get_settings)

):

    """

    Summarize preprocessed scraped job data using an agent.

    

    This endpoint:

    1. Accepts preprocessed scraped data (structured data from backend)

    2. Uses an agent to summarize and structure the information

    3. Returns structured summarized data

    

    Request Body:

        scraped_data: Preprocessed scraped data from backend (structured data)

        openai_api_key: Optional OpenAI API key (uses env var if not provided)

        

    Returns:

        - summarized_data: Structured data from agent summarization

        - success: Whether summarization was successful

        - error: Error message if any

    """

    try:

        from scrapers.response import summarize_scraped_data

        import asyncio

        

        print(f"\n{'='*80}")

        print(f"SUMMARIZER - Processing preprocessed scraped data")

        print(f"{'='*80}")

        

        # Validate scraped_data

        if not request.scraped_data:

            raise ValueError("scraped_data is required and cannot be empty")

        

        # Get OpenAI API key

        openai_key = request.openai_api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if not openai_key:

            raise ValueError("OpenAI API key is required. Provide it in request or set OPENAI_API_KEY environment variable.")

        

        # Use agent to summarize scraped data (run in thread pool since it's sync)

        summarized_data = await asyncio.to_thread(

            summarize_scraped_data,

            request.scraped_data,

            openai_key

        )

        

        print(f"\n{'='*80}")

        print(f"SUCCESS - Summarization completed")

        print(f"{'='*80}")

        

        return SummarizeJobResponse(

            summarized_data=summarized_data,

            success=True,

            error=None

        )

        

    except Exception as e:

        import traceback

        error_msg = str(e)

        traceback.print_exc()

        print(f"\n{'='*80}")

        print(f"ERROR - {error_msg}")

        print(f"{'='*80}")

        

        return SummarizeJobResponse(

            summarized_data={},

            success=False,

            error=error_msg

        )


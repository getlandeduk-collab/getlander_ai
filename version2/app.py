from __future__ import annotations

import asyncio
import json
import os
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
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
)
from utils import decode_base64_pdf, extract_text_from_pdf_bytes, now_iso, make_request_id, redact_long_text, scrape_website_custom
from agents import build_resume_parser, build_scraper, build_scorer, build_summarizer, build_orchestrator
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


# In-memory stores
REQUEST_PROGRESS: Dict[str, ProgressStatus] = {}
SCRAPE_CACHE: Dict[str, Dict[str, Any]] = {}
LAST_REQUESTS_BY_IP: Dict[str, List[float]] = {}


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
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


@app.post("/api/match-jobs", response_model=MatchJobsResponse, dependencies=[Depends(rate_limit)])
async def match_jobs(
    json_body: Optional[str] = Form(default=None),
    pdf_file: Optional[UploadFile] = File(default=None),
    settings: Settings = Depends(get_settings),
):
    request_id = make_request_id()
    REQUEST_PROGRESS[request_id] = ProgressStatus(
        request_id=request_id, status="queued", jobs_total=0, jobs_scraped=0, 
        jobs_cached=0, started_at=now_iso(), updated_at=now_iso()
    )

    try:
        # Parse input - support new format with jobs field, legacy format, and old format
        data: Optional[MatchJobsRequest] = None
        legacy_data: Optional[MatchJobsJsonRequest] = None
        new_format_jobs: Optional[Dict[str, Any]] = None  # New format with jobtitle, joblink, jobdata
        user_id: Optional[str] = None
        
        if json_body:
            try:
                # Handle JSON that might be double-encoded or have extra quotes
                clean_json = json_body.strip()
                if clean_json.startswith('"') and clean_json.endswith('"'):
                    clean_json = clean_json[1:-1].replace('\\"', '"')
                payload = json.loads(clean_json)
                
                # Check for new format with jobs field (jobtitle, joblink, jobdata)
                if "jobs" in payload and isinstance(payload["jobs"], dict):
                    new_format_jobs = payload["jobs"]
                    user_id = payload.get("user_id")
                    print(f"[NEW FORMAT] Detected jobs field with jobtitle, joblink, jobdata")
                # Try new format first (resume + jobs list)
                elif "resume" in payload and "jobs" in payload:
                    data = MatchJobsRequest(**payload)
                else:
                    # Legacy format
                    legacy_data = MatchJobsJsonRequest(**payload)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid JSON body: {e}. Received: {json_body[:100]}"
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

        # STEP 1: Parse Resume
        print("\n" + "="*80)
        print("📄 RESUME PARSER AGENT - Extracting information from OCR text")
        print("="*80)
        
        resume_agent = build_resume_parser(settings.model_name)
        
        resume_prompt = f"""
Extract ALL information from this resume OCR text and return ONLY valid JSON.

Resume text:
{resume_text}

Return this exact structure (no markdown, no explanations):
{{
  "name": "Full name here",
  "email": "email@example.com",
  "phone": "+1234567890",
  "skills": ["Python", "TensorFlow", "Java", etc.],
  "experience_summary": "Brief work history summary",
  "total_years_experience": 1.5,
  "education": [{{"school": "University", "degree": "BS", "dates": "2027"}}],
  "certifications": ["Cert name"],
  "interests": ["Interest 1", "Interest 2"]
}}

Extract every skill, tool, and technology mentioned. Calculate total years from all work experience.
"""
        
        try:
            # Use synchronous run() for phi agents
            resume_response = resume_agent.run(resume_prompt)
            
            # Handle different response types
            if hasattr(resume_response, 'content'):
                response_text = str(resume_response.content)
            elif hasattr(resume_response, 'messages') and resume_response.messages:
                # Get last message content
                last_msg = resume_response.messages[-1]
                response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
            else:
                response_text = str(resume_response)
            
            response_text = response_text.strip()
            
            print("\n[RESUME PARSER RAW OUTPUT]:")
            print(response_text[:800])
            
            # Extract JSON from response
            resume_json = extract_json_from_response(response_text)
            
            # Validate we got something useful
            if not resume_json or not resume_json.get("name") or resume_json.get("name") == "Unknown":
                print("⚠️  Warning: Resume parsing returned incomplete data, attempting fallback extraction")
                
                # Fallback: Extract basic info using regex
                name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', resume_text, re.MULTILINE)
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
                phone_match = re.search(r'\+?\d[\d\s-]{8,}\d', resume_text)
                
                # Extract skills from common keywords
                skill_keywords = [
                    'Python', 'Java', 'C++', 'JavaScript', 'TypeScript', 'React', 'Node',
                    'TensorFlow', 'PyTorch', 'Keras', 'OpenCV', 'SQL', 'MySQL', 'MongoDB',
                    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'Linux',
                    'Machine Learning', 'Deep Learning', 'AI', 'Data Science', 'NLP'
                ]
                found_skills = [skill for skill in skill_keywords if skill.lower() in resume_text.lower()]
                
                resume_json = {
                    "name": name_match.group(1) if name_match else "Unknown Candidate",
                    "email": email_match.group() if email_match else None,
                    "phone": phone_match.group() if phone_match else None,
                    "skills": found_skills or [],
                    "experience_summary": resume_text[:500],
                    "total_years_experience": 1.0,  # Default assumption
                    "education": [],
                    "certifications": [],
                    "interests": []
                }
                print("\n[FALLBACK EXTRACTION]:")
                print(f"Name: {resume_json['name']}")
                print(f"Skills found: {len(resume_json['skills'])}")
                
        except Exception as e:
            print(f"❌ Error parsing resume: {e}")
            import traceback
            print(traceback.format_exc())
            
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
        
        if new_format_jobs:
            # NEW FORMAT: Use jobdata directly with summarizer (skip scraping)
            print("\n" + "="*80)
            print(f"📝 SUMMARIZER - Processing job data directly (new format)")
            print("="*80)
            
            from scrapers.response import summarize_scraped_data
            
            # Extract job information from new format
            job_title = new_format_jobs.get("jobtitle", "Unknown Position")
            job_link = new_format_jobs.get("joblink", "")
            job_data = new_format_jobs.get("jobdata", "")
            
            # Create scraped_data structure for summarizer
            scraped_data = {
                "url": job_link,
                "job_title": job_title,
                "company_name": None,  # Will be extracted by summarizer
                "location": None,
                "description": job_data,
                "qualifications": None,
                "suggested_skills": None,
                "text_content": job_data,
                "html_length": len(job_data)
            }
            
            # Use summarizer to process the job data
            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OpenAI API key is required for summarization")
            
            print(f"Processing job: {job_title}")
            print(f"Job data length: {len(job_data)} characters")
            
            # Run summarizer in thread pool
            summarized_data = await asyncio.to_thread(
                summarize_scraped_data,
                scraped_data,
                openai_key
            )
            
            # Create JobPosting from summarized data
            job = JobPosting(
                url=job_link if job_link else "https://example.com",
                job_title=summarized_data.get("job_title") or job_title,
                company=summarized_data.get("company_name") or "Unknown Company",
                description=summarized_data.get("description") or job_data,
                skills_needed=summarized_data.get("required_skills", []) or [],
                experience_level=summarized_data.get("required_experience"),
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
            else:
                urls = [str(u) for u in legacy_data.urls]
                job_titles = {}
                job_companies = {}
                
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

                        # Clean up extracted values
                        if title:
                            # Remove common suffixes/prefixes
                            title = re.sub(r'\s*[-–—]\s*.+$', '', title)  # Remove " - Company Name"
                            title = re.sub(r'^.+:\s*', '', title)  # Remove "Job Board: "
                            title = title.strip()[:100]  # Limit length
                        
                        if company:
                            company = company.strip()[:50]  # Limit length
                            # Remove common prefixes
                            company = re.sub(r'^at\s+', '', company, flags=re.I)
                            company = company.strip()

                        print(f"\n✓ Scraped {url} ({len(content)} chars)")
                        if title:
                            print(f"  Title extracted: {title}")
                        if company:
                            print(f"  Company extracted: {company}")

                        # Use provided titles/companies first, then fall back to extracted
                        final_title = job_titles.get(url) or title or "Unknown Position"
                        final_company = job_companies.get(url) or company or ''
                        
                        # If still unknown, try to extract from content text using AI or patterns
                        if final_title == "Unknown Position" and content:
                            # Try to find job title pattern in first few lines of content
                            first_lines = content[:500].split('\n')[:5]
                            for line in first_lines:
                                line = line.strip()
                                if line and any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'executive', 'director']):
                                    # Extract potential title (first meaningful line with job keywords)
                                    if len(line) < 100:
                                        final_title = line
                                        break
                        
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

        # STEP 4: Score Jobs
        REQUEST_PROGRESS[request_id].status = "matching"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        
        print("\n" + "="*80)
        print("🤖 JOB SCORER AGENT - Calculating match scores")
        print("="*80)

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
                
                return {
                    "job": job,
                    "match_score": score,
                    "key_matches": data.get("key_matches", []) or [],
                    "requirements_met": int(data.get("requirements_met", 0)),
                    "total_requirements": int(data.get("total_requirements", 1)),
                    "reasoning": data.get("reasoning", "Score calculated based on candidate-job alignment"),
                }
            except Exception as e:
                print(f"❌ Error scoring {job.job_title}: {e}")
                return None

        # Score sequentially to avoid rate limits
        scored = []
        for job in jobs:
            result = score_job_sync(job)
            if result:
                scored.append(result)
            await asyncio.sleep(0.5)  # Rate limit protection

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
            """Generate summary for a matched job."""
            job: JobPosting = entry["job"]
            score = entry["match_score"]
            
            prompt = f"""
Write a 150-200 word unique summary for this job-candidate match.

Candidate: {candidate_profile.name}
- Skills: {', '.join(candidate_profile.skills[:10])}
- Experience: {candidate_profile.total_years_experience} years

Job: {job.job_title} at {job.company}
Match Score: {score:.1%}
Description: {job.description[:1500]}

Explain:
- Why this is {'a strong' if score >= 0.7 else 'a good' if score >= 0.5 else 'a weak'} match
- Specific skills/experience that align
- Growth opportunities
- Important considerations
- Visa sponsorship or scholarship information (if mentioned in the job description)

IMPORTANT: Check the job description for visa sponsorship, visa support, scholarship, H1B, work permit, financial support, or tuition assistance information. 
If found, include it in the summary. If not mentioned, state "No visa sponsorship or scholarship information mentioned."

Be honest about the fit level based on the score.
"""
            try:
                response = summarizer_agent.run(prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    text = str(response.content)
                elif hasattr(response, 'messages') and response.messages:
                    last_msg = response.messages[-1]
                    text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
                else:
                    text = str(response)
                
                text = text.strip()
                
                # Strip markdown code fences if present
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1])
                
                # Truncate at sentence boundary if too long (max 3000 chars, but prefer complete sentences)
                max_length = 3000
                if len(text) > max_length:
                    # Find the last complete sentence before max_length
                    truncated = text[:max_length]
                    # Try to find the last sentence ending
                    last_period = truncated.rfind('.')
                    last_exclamation = truncated.rfind('!')
                    last_question = truncated.rfind('?')
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    
                    if last_sentence_end > max_length * 0.7:  # Only use if we're keeping at least 70% of max
                        text = text[:last_sentence_end + 1]
                    else:
                        # If no good sentence boundary, just truncate at word boundary
                        last_space = truncated.rfind(' ')
                        if last_space > max_length * 0.7:
                            text = text[:last_space] + "..."
                        else:
                            text = truncated + "..."
                
                print(f"✓ Summarized rank {rank}: {job.job_title} ({len(text)} chars)")
                
            except Exception as e:
                print(f"❌ Error summarizing rank {rank}: {e}")
                text = f"Match score: {score:.1%}. {entry.get('reasoning', '')}"
            
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
                scraped_summary=SCRAPE_CACHE.get(str(job.url), {}).get('scraped_summary'),
                visa_scholarship_info=visa_scholarship_info,
            )

        # Generate summaries sequentially
        matched_jobs = []
        for i, entry in enumerate(top_matches):
            result = summarize_sync(entry, i + 1)
            matched_jobs.append(result)
            await asyncio.sleep(0.5)

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

        # Save job applications to Firestore if user_id is provided
        if user_id and matched_jobs:
            try:
                # IMPORTANT: Load environment variables again to ensure they're available
                # This is critical because environment might not be loaded in async context
                load_dotenv()
                load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
                
                # Check environment variable is available (try multiple methods)
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                print(f"\n[DEBUG] GOOGLE_APPLICATION_CREDENTIALS from os.getenv(): {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
                print(f"[DEBUG] GOOGLE_APPLICATION_CREDENTIALS from os.environ: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
                print(f"[DEBUG] Final creds_path: {creds_path}")
                
                from firebase_service import get_firebase_service
                from job_extractor import extract_jobs_from_response
                
                print(f"\n{'='*80}")
                print(f"[SAVE] SAVING JOB APPLICATIONS TO FIRESTORE")
                print(f"{'='*80}")
                print(f"User ID: {user_id}")
                print(f"Number of matched jobs: {len(matched_jobs)}")
                
                # Try to get Firebase service
                print(f"[DEBUG] Attempting to get Firebase service...")
                try:
                    firebase_service = get_firebase_service()
                    print(f"[DEBUG] Firebase service obtained successfully")
                    print(f"[DEBUG] Firebase DB is initialized: {firebase_service._db is not None}")
                except Exception as init_error:
                    print(f"[ERROR] Failed to get Firebase service: {str(init_error)}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise
                
                # Convert MatchedJob Pydantic objects to dictionaries for extraction function
                # This ensures we use the proper formatting with datetime objects
                print(f"[DEBUG] Converting {len(matched_jobs)} matched jobs to API response format...")
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
                            "scraped_summary": job.scraped_summary,
                            "visa_scholarship_info": job.visa_scholarship_info,
                        }
                        for job in matched_jobs
                    ]
                }
                
                print(f"[DEBUG] Extracting and formatting jobs using extract_jobs_from_response...")
                jobs_to_save = extract_jobs_from_response(api_response_format)
                
                # Save all job applications (multiple documents will be created)
                if jobs_to_save:
                    print(f"[INFO] Preparing to save {len(jobs_to_save)} job applications to Firestore...")
                    print(f"[INFO] Each job will be saved as a separate document in users/{user_id}/job_applications/")
                    saved_doc_ids = firebase_service.save_job_applications_batch(user_id, jobs_to_save)
                    print(f"\n{'='*80}")
                    print(f"[SUCCESS] Successfully saved {len(saved_doc_ids)} job applications to Firestore")
                    print(f"[INFO] Document IDs: {saved_doc_ids}")
                    print(f"[INFO] Each document is saved at: users/{user_id}/job_applications/{{doc_id}}")
                    print(f"[PATH] Collection: users/{user_id}/job_applications/")
                    print(f"{'='*80}\n")
                else:
                    print("[WARNING] No job applications to save (extraction returned empty list)")
                    
            except ImportError as e:
                print(f"\n[WARNING] Firebase service not available: {str(e)}")
                print("[INFO] Make sure firebase-admin is installed: pip install firebase-admin")
            except Exception as e:
                print(f"\n[ERROR] Failed to save job applications to Firestore (non-fatal): {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("\n[INFO] Note: The API response will still be returned, but applications were not saved to Firestore.")

        response = MatchJobsResponse(
            candidate_profile=candidate_profile,
            matched_jobs=matched_jobs,
            processing_time="",
            jobs_analyzed=len(jobs),  # Use len(jobs) instead of len(urls) for accuracy
            request_id=request_id,
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
        
        # Convert to Pydantic models
        resumes = [
            FirebaseResume(**resume_data)
            for resume_data in resumes_data
        ]
        
        return FirebaseResumeListResponse(
            user_id=request.user_id,
            resumes=resumes,
            count=len(resumes)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
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

                    description = re.sub(r"^```[\w]*\n", "", description)
                    description = re.sub(r"\n```$", "", description)
                    description = description.strip()

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

        job_info["visa_scholarship_info"] = visa_info or "Not specified"
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
            """Synchronous Playwright scraping function."""
            import re
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                
                # Wait for page to load
                page.wait_for_load_state("networkidle", timeout=30000)
                
                # Get page title
                page_title = page.title()
                current_url = page.url
                html_content = page.content()
                text_content = page.inner_text("body")
                
                # Extract job title
                job_title = None
                title_selectors = [
                    '.jobs-details-top-card__job-title',
                    'h1[data-test-id*="job-title"]',
                    '.topcard__title',
                    'h1',
                    '.job-title'
                ]
                for selector in title_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            job_title = element.inner_text().strip()
                            break
                    except:
                        continue
                
                # Extract company name
                company_name = None
                company_selectors = [
                    '.jobs-details-top-card__company-name',
                    '[data-test-id*="company"]',
                    '.topcard__org-name',
                    '.company-name',
                    'a[href*="/company/"]',
                    '.jobs-details-top-card__company-info'
                ]
                for selector in company_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            company_name = element.inner_text().strip()
                            break
                    except:
                        continue
                
                if not company_name and job_title:
                    company_match = re.search(rf'{re.escape(job_title)}\n([A-Za-z0-9\s&]+)\s+([A-Za-z,\s]+,\s*[A-Za-z,\s]+)', text_content)
                    if company_match:
                        company_name = company_match.group(1).strip()
                
                # Extract job description
                description = None
                desc_selectors = [
                    '.jobs-description__text',
                    '.jobs-box__html-content',
                    '[data-test-id*="description"]',
                    '.job-description',
                    '#job-description',
                    '.jobs-description-content__text',
                    'div[data-test-id*="job-details"]'
                ]
                for selector in desc_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            description = element.inner_text().strip()
                            break
                    except:
                        continue
                
                if not description:
                    desc_match = re.search(r'Job Description\s*\n\n(.*?)(?:\n\nAdditional Information|\n\nShow more|\Z)', text_content, re.DOTALL)
                    if desc_match:
                        description = desc_match.group(1).strip()
                    else:
                        desc_match = re.search(r'Job Description\s*\n\n(.*)', text_content, re.DOTALL)
                        if desc_match:
                            description = desc_match.group(1).strip()
                            description = re.split(r'\n\nSimilar jobs|\n\nReferrals', description)[0]
                
                # Extract location
                location = None
                location_selectors = [
                    '.jobs-details-top-card__primary-description-without-tagline',
                    '.jobs-details-top-card__bullet',
                    '[data-test-id*="location"]'
                ]
                for selector in location_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            location = element.inner_text().strip()
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
                
                scraped_data = {
                    "url": current_url,
                    "title": page_title,
                    "job_title": job_title,
                    "company_name": company_name,
                    "location": location,
                    "description": description,
                    "qualifications": qualifications,
                    "suggested_skills": skills,
                    "text_content": text_content,
                    "html_length": len(html_content)
                }
                
                browser.close()
            
            return scraped_data
        
        # Run Playwright scraping in thread pool
        scraped_data = await asyncio.to_thread(scrape_with_playwright, url)
        
        print(f"\n{'='*80}")
        print(f"AGENT SUMMARIZATION - Processing scraped data")
        print(f"{'='*80}")
        
        # STEP 4: Summarize scraped data
        openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        summarized_data = await asyncio.to_thread(summarize_scraped_data, scraped_data, openai_key)
        
        # STEP 5: Create JobPosting from summarized data
        job = JobPosting(
            url=url,
            job_title=summarized_data.get("job_title") or scraped_data.get("job_title") or "Unknown Position",
            company=summarized_data.get("company_name") or scraped_data.get("company_name") or "Unknown Company",
            description=summarized_data.get("description") or scraped_data.get("description") or scraped_data.get("text_content", "")[:2000],
            skills_needed=summarized_data.get("required_skills", []) or [],
            experience_level=summarized_data.get("required_experience"),
            salary=summarized_data.get("salary")
        )
        
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
                from job_extractor import detect_portal

                firebase_service = get_firebase_service()

                portal = detect_portal(url)
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
        # Extract visa/scholarship info from summarized_data
        visa_scholarship_info = summarized_data.get("visa_scholarship_info") or "Not specified"
        
        return PlaywrightScrapeResponse(
            url=url,
            scraped_data=scraped_data,
            summarized_data=summarized_data,
            match_score=scoring_result["match_score"] if scoring_result else None,
            key_matches=scoring_result["key_matches"] if scoring_result else None,
            requirements_met=scoring_result["requirements_met"] if scoring_result else None,
            total_requirements=scoring_result["total_requirements"] if scoring_result else None,
            reasoning=scoring_result["reasoning"] if scoring_result else None,
            visa_scholarship_info=visa_scholarship_info,
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
            match_score=None,
            key_matches=None,
            requirements_met=None,
            total_requirements=None,
            reasoning=None,
            visa_scholarship_info="Not specified",
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

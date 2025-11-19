"""
Sponsorship checking utility for matching companies against UK visa sponsorship database.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    try:
        from rapidfuzz import fuzz, process
        FUZZYWUZZY_AVAILABLE = True
    except ImportError:
        FUZZYWUZZY_AVAILABLE = False


# CSV file path
CSV_PATH = Path(__file__).resolve().parent / "2025-11-07_-_Worker_and_Temporary_Worker.csv"

# Cache for loaded CSV data
_sponsorship_df: Optional[pd.DataFrame] = None


def load_sponsorship_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Load sponsorship CSV data with caching.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with sponsorship data
    """
    global _sponsorship_df
    
    if _sponsorship_df is not None:
        return _sponsorship_df
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Sponsorship CSV file not found: {csv_path}")
    
    try:
        _sponsorship_df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"[Sponsorship] Loaded {len(_sponsorship_df)} companies from CSV")
        return _sponsorship_df
    except Exception as e:
        raise RuntimeError(f"Failed to load sponsorship CSV: {str(e)}")


def clean_company_name(name: str) -> str:
    """
    Clean company name by removing legal suffixes and normalizing.
    
    Args:
        name: Raw company name
        
    Returns:
        Cleaned company name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Remove quotes and extra whitespace
    name = name.strip().strip('"').strip("'")
    
    # Remove common legal suffixes (case-insensitive)
    suffixes = [
        r'\s+inc\.?$', r'\s+incorporated$',
        r'\s+ltd\.?$', r'\s+limited$',
        r'\s+llc\.?$', r'\s+ll\.?c\.?$',
        r'\s+corp\.?$', r'\s+corporation$',
        r'\s+plc\.?$', r'\s+public limited company$',
        r'\s+llp\.?$', r'\s+limited liability partnership$',
        r'\s+p\.?c\.?$', r'\s+professional corporation$',
        r'\s+co\.?$', r'\s+company$',
        r'\s+group$', r'\s+holdings?$',
    ]
    
    for suffix_pattern in suffixes:
        name = re.sub(suffix_pattern, '', name, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def extract_company_name(job_content: str, job_company: Optional[str] = None) -> Optional[str]:
    """
    Extract company name from job content or use provided company name.
    
    Args:
        job_content: Scraped job posting content
        job_company: Pre-extracted company name (if available)
        
    Returns:
        Extracted/cleaned company name or None
    """
    # If company already provided, clean and return it
    if job_company:
        cleaned = clean_company_name(job_company)
        if cleaned:
            return cleaned
    
    if not job_content:
        return None
    
    # Try to extract company name from content using regex patterns
    patterns = [
        r'company[:\s]+([A-Z][A-Za-z0-9\s&.,-]{2,50})',
        r'employer[:\s]+([A-Z][A-Za-z0-9\s&.,-]{2,50})',
        r'organization[:\s]+([A-Z][A-Za-z0-9\s&.,-]{2,50})',
        r'at\s+([A-Z][A-Za-z0-9\s&.,-]{2,50})\s+(?:Ltd|Limited|Inc|LLC|Corp)',
        r'([A-Z][A-Za-z0-9\s&.,-]{2,50})\s+(?:Ltd|Limited|Inc|LLC|Corp)',
    ]
    
    content_lower = job_content.lower()
    for pattern in patterns:
        matches = re.finditer(pattern, job_content, re.IGNORECASE)
        for match in matches:
            company = match.group(1).strip()
            # Validate it looks like a company name
            if len(company) >= 3 and len(company) <= 50:
                # Remove common prefixes
                company = re.sub(r'^(the|a|an)\s+', '', company, flags=re.IGNORECASE)
                cleaned = clean_company_name(company)
                if cleaned:
                    return cleaned
    
    return None


def find_company_in_csv(company_name: str, df: pd.DataFrame, threshold: int = 80) -> Optional[Dict[str, Any]]:
    """
    Find company in CSV using fuzzy matching with multiple strategies.
    
    Args:
        company_name: Company name to search for
        df: DataFrame with sponsorship data
        threshold: Minimum similarity score (0-100) - lowered to 80 for better matching
        
    Returns:
        Dictionary with company info if found, None otherwise
    """
    if not company_name or not FUZZYWUZZY_AVAILABLE:
        print(f"[Sponsorship] Fuzzy matching not available or company name is empty")
        return None
    
    cleaned_name = clean_company_name(company_name)
    if not cleaned_name:
        print(f"[Sponsorship] Company name could not be cleaned: {company_name}")
        return None
    
    print(f"[Sponsorship] Searching for company: '{company_name}' (cleaned: '{cleaned_name}')")
    
    # Get all organization names from CSV
    org_names = df['Organisation Name'].astype(str).tolist()
    
    # Use fuzzy matching to find best match with multiple strategies
    try:
        # Strategy 1: Token sort ratio (handles word order differences)
        best_match_1, score_1 = process.extractOne(
            cleaned_name,
            org_names,
            scorer=fuzz.token_sort_ratio
        )
        
        # Strategy 2: Partial ratio (handles substring matches like "UK" vs "United Kingdom")
        best_match_2, score_2 = process.extractOne(
            cleaned_name,
            org_names,
            scorer=fuzz.partial_ratio
        )
        
        # Strategy 3: Token set ratio (handles duplicates and word order)
        best_match_3, score_3 = process.extractOne(
            cleaned_name,
            org_names,
            scorer=fuzz.token_set_ratio
        )
        
        # Strategy 4: WRatio (weighted combination of all methods)
        best_match_4, score_4 = process.extractOne(
            cleaned_name,
            org_names,
            scorer=fuzz.WRatio
        )
        
        # Find the best match across all strategies
        matches = [
            (best_match_1, score_1, "token_sort"),
            (best_match_2, score_2, "partial"),
            (best_match_3, score_3, "token_set"),
            (best_match_4, score_4, "WRatio")
        ]
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score, strategy = matches[0]
        
        print(f"[Sponsorship] Best match: '{best_match}' (score: {best_score}%, strategy: {strategy})")
        print(f"[Sponsorship] All strategies: token_sort={score_1}%, partial={score_2}%, token_set={score_3}%, WRatio={score_4}%")
        
        # Use a lower threshold for partial matches (handles "UK" vs "United Kingdom")
        effective_threshold = threshold
        if strategy == "partial" and "uk" in cleaned_name.lower() and "united kingdom" in best_match.lower():
            # Special case: "UK" vs "United Kingdom" should match with lower threshold
            effective_threshold = max(70, threshold - 10)
            print(f"[Sponsorship] Using lower threshold ({effective_threshold}%) for UK/United Kingdom match")
        
        if best_score >= effective_threshold:
            # Get all rows for this company (may have multiple routes)
            company_rows = df[df['Organisation Name'] == best_match]
            
            # Aggregate information
            routes = company_rows['Route'].dropna().unique().tolist()
            types = company_rows['Type & Rating'].dropna().unique().tolist()
            locations = company_rows[['Town/City', 'County']].dropna()
            
            location_str = ""
            if not locations.empty:
                loc_parts = []
                for _, row in locations.iterrows():
                    parts = [str(row['Town/City']), str(row['County'])]
                    loc_str = ", ".join([p for p in parts if p and p != 'nan'])
                    if loc_str:
                        loc_parts.append(loc_str)
                if loc_parts:
                    location_str = "; ".join(set(loc_parts))[:200]
            
            # Build summary with location information
            summary_parts = [f"{best_match} is a registered UK visa sponsor."]
            if routes:
                summary_parts.append(f"Visa Routes: {', '.join(routes)}.")
            if types:
                summary_parts.append(f"Worker Types: {', '.join(types)}.")
            if location_str:
                summary_parts.append(f"Location(s): {location_str}.")
            
            summary = " ".join(summary_parts)
            
            print(f"[Sponsorship] ✓ Match found! Score: {best_score}%")
            print(f"[Sponsorship] Routes: {', '.join(routes) if routes else 'None'}")
            print(f"[Sponsorship] Locations: {location_str if location_str else 'None'}")
            
            return {
                'company_name': best_match,
                'match_score': best_score,
                'sponsors_workers': True,
                'visa_types': ", ".join(routes) if routes else "Not specified",
                'worker_types': ", ".join(types) if types else "Not specified",
                'locations': location_str,
                'total_listings': len(company_rows),
                'summary': summary
            }
    except Exception as e:
        print(f"[Sponsorship] Error in fuzzy matching: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    
    # If we get here, no match was found
    print(f"[Sponsorship] ✗ No match found above threshold ({threshold}%)")
    return None


def check_sponsorship(company_name: Optional[str], job_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if a company sponsors workers by looking it up in the CSV.
    
    Args:
        company_name: Company name (if already extracted)
        job_content: Job posting content (for extraction if company_name not provided)
        
    Returns:
        Dictionary with sponsorship information
    """
    try:
        # Extract company name if not provided
        if not company_name:
            company_name = extract_company_name(job_content or "")
        
        if not company_name:
            return {
                'company_name': None,
                'sponsors_workers': False,
                'visa_types': None,
                'summary': 'Company name could not be extracted from job posting.'
            }
        
        # Load CSV data
        df = load_sponsorship_data()
        
        # Search for company (using lower threshold for better matching)
        match_result = find_company_in_csv(company_name, df, threshold=80)
        
        if match_result:
            # Use summary from match_result if available, otherwise build one
            summary = match_result.get('summary')
            if not summary:
                summary_parts = [
                    f"{match_result['company_name']} is a registered UK visa sponsor.",
                    f"Worker Types: {match_result['worker_types']}",
                    f"Visa Routes: {match_result['visa_types']}",
                ]
                if match_result.get('locations'):
                    summary_parts.append(f"Locations: {match_result['locations']}")
                summary_parts.append(f"Total active listings: {match_result['total_listings']}")
                summary = " ".join(summary_parts)
            
            return {
                'company_name': match_result['company_name'],
                'sponsors_workers': True,
                'visa_types': match_result['visa_types'],
                'summary': summary
            }
        else:
            return {
                'company_name': company_name,
                'sponsors_workers': False,
                'visa_types': None,
                'summary': f"{company_name} was not found in the UK visa sponsorship database. This may mean they do not currently sponsor workers, or the company name does not match exactly."
            }
            
    except FileNotFoundError as e:
        return {
            'company_name': company_name or "Unknown",
            'sponsors_workers': False,
            'visa_types': None,
            'summary': f'Sponsorship database not available: {str(e)}'
        }
    except Exception as e:
        print(f"[Sponsorship] Error checking sponsorship: {e}")
        return {
            'company_name': company_name or "Unknown",
            'sponsors_workers': False,
            'visa_types': None,
            'summary': f'Error checking sponsorship: {str(e)}'
        }


def get_company_info_from_web(company_name: str, openai_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get company information from the web using Phi agent with DuckDuckGo search.
    
    Args:
        company_name: Company name to search for
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        Company information string or None if error occurs
    """
    if not company_name:
        return None
    
    try:
        from phi.agent import Agent
        from phi.model.openai import OpenAIChat
        from phi.tools.duckduckgo import DuckDuckGo
        
        # Get API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"[Company Info] OpenAI API key not available, skipping web search")
            return None
        
        print(f"[Company Info] Fetching company information for: {company_name}")
        
        # Create web agent
        web_agent = Agent(
            name="Company Info Agent",
            model=OpenAIChat(id="gpt-4o", api_key=api_key),
            tools=[DuckDuckGo()],
            instructions=[
                "Search for information about the company including:",
                "- Company overview and industry",
                "- Company size and headquarters location",
                "- Recent news or developments",
                "- Company culture and values (if available)",
                "Always include sources in your response.",
                "Keep the response concise and informative (2-3 paragraphs maximum)."
            ],
            show_tool_calls=False,
            markdown=False,
        )
        
        # Search query
        query = f"Tell me about {company_name} company: overview, industry, size, headquarters, and recent news"
        
        # Get response (non-streaming for simplicity)
        response = web_agent.run(query, stream=False)
        
        # Extract content from RunResponse
        company_info = None
        if hasattr(response, 'content'):
            company_info = response.content
            # Convert to string if it's not already
            if not isinstance(company_info, str):
                company_info = str(company_info)
        elif isinstance(response, str):
            company_info = response
        else:
            # Try to get text representation
            company_info = str(response)
        
        if company_info and len(company_info.strip()) > 0:
            print(f"[Company Info] Successfully fetched company information ({len(company_info)} characters)")
            return company_info
        else:
            print(f"[Company Info] No company information returned from web search")
            return None
        
    except ImportError as e:
        print(f"[Company Info] Phi agent dependencies not available: {e}")
        print(f"[Company Info] Install phidata: pip install phidata")
        return None
    except Exception as e:
        print(f"[Company Info] Error fetching company information: {e}")
        import traceback
        print(traceback.format_exc())
        return None


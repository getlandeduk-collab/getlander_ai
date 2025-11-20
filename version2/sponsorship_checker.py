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


def find_multiple_company_matches_in_csv(company_name: str, df: pd.DataFrame, threshold: int = 70, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Find multiple company matches in CSV using fuzzy matching with multiple strategies.
    Returns top N matches sorted by score.
    
    Args:
        company_name: Company name to search for
        df: DataFrame with sponsorship data
        threshold: Minimum similarity score (0-100) - lowered to 70 to get more candidates
        top_n: Number of top matches to return
        
    Returns:
        List of dictionaries with company info, sorted by match score (highest first)
    """
    if not company_name or not FUZZYWUZZY_AVAILABLE:
        print(f"[Sponsorship] Fuzzy matching not available or company name is empty")
        return []
    
    cleaned_name = clean_company_name(company_name)
    if not cleaned_name:
        print(f"[Sponsorship] Company name could not be cleaned: {company_name}")
        return []
    
    print(f"[Sponsorship] Searching for multiple matches for company: '{company_name}' (cleaned: '{cleaned_name}')")
    
    # Get all organization names from CSV
    org_names = df['Organisation Name'].astype(str).tolist()
    
    # Collect all unique matches from different strategies
    all_matches = {}
    
    try:
        # Strategy 1: Token sort ratio (handles word order differences)
        matches_1 = process.extract(cleaned_name, org_names, scorer=fuzz.token_sort_ratio, limit=top_n)
        for match, score in matches_1:
            if match not in all_matches or all_matches[match]['max_score'] < score:
                all_matches[match] = {
                    'company_name': match,
                    'max_score': score,
                    'token_sort_score': score,
                    'strategy': 'token_sort'
                }
        
        # Strategy 2: Partial ratio (handles substring matches)
        matches_2 = process.extract(cleaned_name, org_names, scorer=fuzz.partial_ratio, limit=top_n)
        for match, score in matches_2:
            if match not in all_matches:
                all_matches[match] = {
                    'company_name': match,
                    'max_score': score,
                    'partial_score': score,
                    'strategy': 'partial'
                }
            else:
                all_matches[match]['partial_score'] = score
                if score > all_matches[match]['max_score']:
                    all_matches[match]['max_score'] = score
                    all_matches[match]['strategy'] = 'partial'
        
        # Strategy 3: Token set ratio (handles duplicates and word order)
        matches_3 = process.extract(cleaned_name, org_names, scorer=fuzz.token_set_ratio, limit=top_n)
        for match, score in matches_3:
            if match not in all_matches:
                all_matches[match] = {
                    'company_name': match,
                    'max_score': score,
                    'token_set_score': score,
                    'strategy': 'token_set'
                }
            else:
                all_matches[match]['token_set_score'] = score
                if score > all_matches[match]['max_score']:
                    all_matches[match]['max_score'] = score
                    all_matches[match]['strategy'] = 'token_set'
        
        # Strategy 4: WRatio (weighted combination of all methods)
        matches_4 = process.extract(cleaned_name, org_names, scorer=fuzz.WRatio, limit=top_n)
        for match, score in matches_4:
            if match not in all_matches:
                all_matches[match] = {
                    'company_name': match,
                    'max_score': score,
                    'wratio_score': score,
                    'strategy': 'WRatio'
                }
            else:
                all_matches[match]['wratio_score'] = score
                if score > all_matches[match]['max_score']:
                    all_matches[match]['max_score'] = score
                    all_matches[match]['strategy'] = 'WRatio'
        
        # Convert to list and filter by threshold
        candidate_matches = [
            match_data for match_data in all_matches.values()
            if match_data['max_score'] >= threshold
        ]
        
        # Sort by max_score descending
        candidate_matches.sort(key=lambda x: x['max_score'], reverse=True)
        
        # Take top N
        top_matches = candidate_matches[:top_n]
        
        print(f"[Sponsorship] Found {len(top_matches)} candidate matches above threshold ({threshold}%)")
        for i, match in enumerate(top_matches, 1):
            print(f"[Sponsorship]   {i}. {match['company_name']} (score: {match['max_score']}%, strategy: {match['strategy']})")
        
        # Build detailed match info for each candidate
        result_matches = []
        for match_data in top_matches:
            company_name_match = match_data['company_name']
            company_rows = df[df['Organisation Name'] == company_name_match]
            
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
            
            result_matches.append({
                'company_name': company_name_match,
                'match_score': match_data['max_score'],
                'strategy': match_data['strategy'],
                'sponsors_workers': True,
                'visa_types': ", ".join(routes) if routes else "Not specified",
                'worker_types': ", ".join(types) if types else "Not specified",
                'locations': location_str,
                'total_listings': len(company_rows),
                'token_sort_score': match_data.get('token_sort_score', 0),
                'partial_score': match_data.get('partial_score', 0),
                'token_set_score': match_data.get('token_set_score', 0),
                'wratio_score': match_data.get('wratio_score', 0),
            })
        
        return result_matches
        
    except Exception as e:
        print(f"[Sponsorship] Error in fuzzy matching: {e}")
        import traceback
        print(traceback.format_exc())
        return []
    
    # If we get here, no matches were found
    print(f"[Sponsorship] ✗ No matches found above threshold ({threshold}%)")
    return []


def find_company_in_csv(company_name: str, df: pd.DataFrame, threshold: int = 80) -> Optional[Dict[str, Any]]:
    """
    Find company in CSV using fuzzy matching with multiple strategies.
    This is the legacy function that returns a single best match.
    For better accuracy, use find_multiple_company_matches_in_csv with select_correct_company_match.
    
    Args:
        company_name: Company name to search for
        df: DataFrame with sponsorship data
        threshold: Minimum similarity score (0-100) - lowered to 80 for better matching
        
    Returns:
        Dictionary with company info if found, None otherwise
    """
    matches = find_multiple_company_matches_in_csv(company_name, df, threshold=threshold, top_n=1)
    if matches:
        match = matches[0]
        # Build summary
        summary_parts = [f"{match['company_name']} is a registered UK visa sponsor."]
        if match.get('visa_types') and match['visa_types'] != "Not specified":
            summary_parts.append(f"Visa Routes: {match['visa_types']}.")
        if match.get('worker_types') and match['worker_types'] != "Not specified":
            summary_parts.append(f"Worker Types: {match['worker_types']}.")
        if match.get('locations'):
            summary_parts.append(f"Location(s): {match['locations']}.")
        
        match['summary'] = " ".join(summary_parts)
        return match
    return None


def _extract_location_from_job_content(job_content: Optional[str]) -> Optional[str]:
    """
    Extract location information from job posting content.
    
    Args:
        job_content: Job posting content text
        
    Returns:
        Extracted location string or None
    """
    if not job_content:
        return None
    
    import re
    
    # Common location patterns
    location_patterns = [
        r'location[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\.|\n|$)',
        r'based[:\s]+(?:in|at|near)[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\.|\n|$)',
        r'office[:\s]+(?:in|at|located)[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\.|\n|$)',
        r'(?:located|situated|headquartered)[:\s]+(?:in|at|near)[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\.|\n|$)',
        r'(London|Manchester|Birmingham|Leeds|Glasgow|Edinburgh|Liverpool|Bristol|Cardiff|Belfast|Newcastle|Sheffield|Nottingham|Leicester|Coventry|Brighton|Oxford|Cambridge)',
    ]
    
    content_lower = job_content.lower()
    for pattern in location_patterns:
        matches = re.finditer(pattern, job_content, re.IGNORECASE)
        for match in matches:
            location = match.group(1).strip() if match.groups() else match.group(0).strip()
            # Clean up location
            location = re.sub(r'[.,;]$', '', location)  # Remove trailing punctuation
            location = location[:100]  # Limit length
            if location and len(location) >= 2:
                return location
    
    return None


def select_correct_company_match(
    job_company_name: str,
    candidate_matches: List[Dict[str, Any]],
    job_content: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Use a Phidata agent to determine which candidate match is the correct company.
    Uses location information from both CSV matches and job content to improve accuracy.
    
    Args:
        job_company_name: The company name from the job posting
        candidate_matches: List of candidate matches from CSV (from find_multiple_company_matches_in_csv)
        job_content: Optional job posting content for additional context
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        The selected match dictionary, or None if no match is selected
    """
    if not candidate_matches:
        return None
    
    # If only one match, return it (no need for agent)
    if len(candidate_matches) == 1:
        print(f"[Sponsorship] Only one candidate match found, using it directly")
        return candidate_matches[0]
    
    try:
        from phi.agent import Agent
        from phi.model.openai import OpenAIChat
        
        # Get API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"[Sponsorship] OpenAI API key not available, using highest scoring match")
            return candidate_matches[0]  # Fallback to highest score
        
        print(f"[Sponsorship] Using AI agent to select correct company from {len(candidate_matches)} candidates")
        
        # Extract location from job content
        job_location = _extract_location_from_job_content(job_content)
        if job_location:
            print(f"[Sponsorship] Extracted location from job posting: {job_location}")
        
        # Build candidate list for the agent with enhanced location info
        candidate_list = []
        for i, match in enumerate(candidate_matches, 1):
            candidate_info = f"{i}. {match['company_name']} (Match Score: {match['match_score']}%)"
            
            # Emphasize location information from CSV
            if match.get('locations'):
                candidate_info += f"\n   Location(s): {match['locations']}"
            
            if match.get('visa_types') and match['visa_types'] != "Not specified":
                candidate_info += f"\n   Visa Routes: {match['visa_types']}"
            
            if match.get('worker_types') and match['worker_types'] != "Not specified":
                candidate_info += f"\n   Worker Types: {match['worker_types']}"
            
            candidate_list.append(candidate_info)
        
        candidates_text = "\n".join(candidate_list)
        
        # Build context from job content (first 500 chars to avoid token limits)
        job_context = ""
        if job_content:
            job_context = f"\n\nJob Posting Context (first 500 chars):\n{job_content[:500]}"
        
        # Add location context if extracted
        location_context = ""
        if job_location:
            location_context = f"\n\nJob Location: {job_location}\n(Use this to match against candidate company locations from the CSV database)"
        
        # Create selection agent
        selection_agent = Agent(
            name="Company Match Selector",
            model=OpenAIChat(id="gpt-4o", api_key=api_key),
            instructions=[
                "You are an expert at matching company names. Your task is to determine which candidate company",
                "from the UK visa sponsorship database is the correct match for the job posting company.",
                "",
                "Consider the following factors (in order of importance):",
                "1. Exact name matches are preferred",
                "2. Location matching: Match the job location with the candidate company's locations from the CSV database",
                "   - Location matches are STRONG indicators of correctness",
                "   - Consider city names, counties, and regions",
                "   - Even partial location matches can help identify the correct company",
                "3. Common abbreviations (e.g., 'Ltd' vs 'Limited', 'Inc' vs 'Incorporated')",
                "4. Company name variations and legal entity suffixes",
                "5. Industry and business context from the job posting",
                "",
                "CRITICAL: Use location information from both the job posting AND the CSV database to help identify",
                "the correct match. Companies often have specific locations where they operate.",
                "",
                "You must respond with ONLY the number (1, 2, 3, etc.) corresponding to the correct match.",
                "If none of the candidates seem correct, respond with '0'.",
                "Do not include any explanation, just the number.",
            ],
            show_tool_calls=False,
            markdown=False,
        )
        
        # Build prompt
        prompt = f"""Given the job posting company name "{job_company_name}", which of these candidate companies from the UK visa sponsorship database is the correct match?
{location_context}
Candidates from CSV Database:
{candidates_text}
{job_context}

Respond with ONLY the number (1-{len(candidate_matches)}) of the correct match, or 0 if none match."""
        
        # Get response
        response = selection_agent.run(prompt, stream=False)
        
        # Extract content from RunResponse
        response_text = None
        if hasattr(response, 'content'):
            response_text = response.content
            if not isinstance(response_text, str):
                response_text = str(response_text)
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Parse the selected number
        import re
        match = re.search(r'\b([0-9]+)\b', response_text.strip())
        if match:
            selected_num = int(match.group(1))
            if 1 <= selected_num <= len(candidate_matches):
                selected_match = candidate_matches[selected_num - 1]
                print(f"[Sponsorship] ✓ AI agent selected: {selected_match['company_name']} (option {selected_num})")
                return selected_match
            elif selected_num == 0:
                print(f"[Sponsorship] ✗ AI agent determined none of the candidates match")
                return None
            else:
                print(f"[Sponsorship] ⚠️  AI agent returned invalid number ({selected_num}), using highest scoring match")
                return candidate_matches[0]
        else:
            print(f"[Sponsorship] ⚠️  Could not parse AI agent response: {response_text[:100]}, using highest scoring match")
            return candidate_matches[0]
            
    except ImportError as e:
        print(f"[Sponsorship] Phi agent dependencies not available: {e}, using highest scoring match")
        return candidate_matches[0]
    except Exception as e:
        print(f"[Sponsorship] Error in AI agent selection: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback to highest scoring match
        return candidate_matches[0]


def check_sponsorship(company_name: Optional[str], job_content: Optional[str] = None, openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if a company sponsors workers by looking it up in the CSV.
    Uses multiple candidate matches and an AI agent to select the correct company.
    
    Args:
        company_name: Company name (if already extracted)
        job_content: Job posting content (for extraction if company_name not provided)
        openai_api_key: OpenAI API key for AI agent selection (optional)
        
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
        
        # Find multiple candidate matches (using lower threshold to get more candidates)
        candidate_matches = find_multiple_company_matches_in_csv(company_name, df, threshold=70, top_n=5)
        
        if candidate_matches:
            # Use AI agent to select the correct match
            selected_match = select_correct_company_match(
                company_name,
                candidate_matches,
                job_content,
                openai_api_key
            )
            
            if selected_match:
                # Build summary
                summary_parts = [f"{selected_match['company_name']} is a registered UK visa sponsor."]
                if selected_match.get('visa_types') and selected_match['visa_types'] != "Not specified":
                    summary_parts.append(f"Visa Routes: {selected_match['visa_types']}.")
                if selected_match.get('worker_types') and selected_match['worker_types'] != "Not specified":
                    summary_parts.append(f"Worker Types: {selected_match['worker_types']}.")
                if selected_match.get('locations'):
                    summary_parts.append(f"Location(s): {selected_match['locations']}.")
                summary_parts.append(f"Total active listings: {selected_match['total_listings']}.")
                
                summary = " ".join(summary_parts)
                
                print(f"[Sponsorship] ✓ Final selected match: {selected_match['company_name']} (score: {selected_match['match_score']}%)")
                
                return {
                    'company_name': selected_match['company_name'],
                    'sponsors_workers': True,
                    'visa_types': selected_match['visa_types'],
                    'summary': summary,
                    'found_in_csv': True
                }
            else:
                # Agent determined none of the candidates match
                print(f"[Sponsorship] ✗ AI agent determined none of the {len(candidate_matches)} candidates are correct")
                return {
                    'company_name': company_name,
                    'sponsors_workers': False,
                    'visa_types': None,
                    'summary': f"{company_name} was not found in the UK visa sponsorship database. The AI agent reviewed {len(candidate_matches)} similar company names but determined none match. This may mean they do not currently sponsor workers, or the company name does not match exactly.",
                    'found_in_csv': False
                }
        else:
            # No candidate matches found in CSV
            print(f"[Sponsorship] ✗ No candidate matches found above threshold")
            return {
                'company_name': company_name,
                'sponsors_workers': False,
                'visa_types': None,
                'summary': f"{company_name} was not found in the UK visa sponsorship database. This may mean they do not currently sponsor workers, or the company name does not match exactly.",
                'found_in_csv': False
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
        import traceback
        print(traceback.format_exc())
        return {
            'company_name': company_name or "Unknown",
            'sponsors_workers': False,
            'visa_types': None,
            'summary': f'Error checking sponsorship: {str(e)}'
        }


def get_company_info_from_web(company_name: str, openai_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get company information from the web using Phi agent with DuckDuckGo search.
    Also searches specifically for visa sponsorship information.
    
    Args:
        company_name: Company name to search for
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        Company information string including sponsorship availability, or None if error occurs
    """
    if not company_name:
        return None
    
    try:
        from phi.agent import Agent
        from phi.model.openai import OpenAIChat
        # Use ddgs instead of duckduckgo_search to avoid warnings
        try:
            from phi.tools.duckduckgo import DuckDuckGo
        except ImportError:
            # Fallback: try to use ddgs directly if phi tools not available
            print(f"[Company Info] DuckDuckGo tool not available, trying direct search")
            return _get_company_info_direct(company_name)
        
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
                f"Search for comprehensive information about {company_name} including:",
                "- Company overview, industry, and what they do",
                "- Company size, headquarters location, and global presence",
                "- UK visa sponsorship availability and policies (CRITICAL - search specifically for this)",
                "- Recent news or developments",
                "- Company culture and values (if available)",
                "",
                "IMPORTANT: Specifically search for information about UK visa sponsorship, work visa sponsorship, or skilled worker visa sponsorship for this company.",
                "If sponsorship information is found, clearly state whether they sponsor UK work visas.",
                "Always include sources in your response.",
                "Keep the response concise and informative (2-3 paragraphs maximum)."
            ],
            show_tool_calls=False,
            markdown=False,
        )
        
        # Search query - include sponsorship search
        query = f"Tell me about {company_name} company: overview, industry, size, headquarters, UK visa sponsorship availability, and recent news"
        
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
        return _get_company_info_direct(company_name)
    except Exception as e:
        print(f"[Company Info] Error fetching company information: {e}")
        import traceback
        print(traceback.format_exc())
        return _get_company_info_direct(company_name)


def _get_company_info_direct(company_name: str) -> Optional[str]:
    """Fallback method using ddgs directly if Phi tools are not available."""
    try:
        from ddgs import DDGS
        
        print(f"[Company Info] Using direct DuckDuckGo search for: {company_name}")
        with DDGS() as ddgs:
            # Search for company info
            results = list(ddgs.text(f"{company_name} company UK visa sponsorship", max_results=3))
            
            if results:
                info_parts = []
                for result in results:
                    info_parts.append(f"{result.get('title', '')}: {result.get('body', '')}")
                
                company_info = "\n\n".join(info_parts)
                print(f"[Company Info] Successfully fetched company information via direct search ({len(company_info)} characters)")
                return company_info
            else:
                print(f"[Company Info] No results found via direct search")
                return None
    except ImportError:
        print(f"[Company Info] ddgs package not available. Install with: pip install ddgs")
        return None
    except Exception as e:
        print(f"[Company Info] Error in direct search: {e}")
        return None


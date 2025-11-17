"""
Agent-based summarization of scraped job data.
Takes scraped_data from playwright_scraper and returns structured information.
"""
from typing import Dict, Any, Optional, List
import os
import re
import json
from phi.agent import Agent
from phi.model.openai import OpenAIChat


def summarize_scraped_data(
    scraped_data: Dict[str, Any],
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use an agent to summarize scraped job data into structured format.
    
    Args:
        scraped_data: Dictionary containing scraped job information from playwright_scraper
        openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
    
    Returns:
        Dictionary with structured job information:
        - job_title
        - company_name
        - location
        - description
        - required_skills
        - required_experience
        - qualifications
        - responsibilities
        - salary
        - job_type
        - suggested_skills
    """
    # Set OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
    
    # Create agent
    agent = Agent(
        show_tool_calls=True,
        markdown=True,
        model=OpenAIChat(id="gpt-4o-mini")
    )
    
    # Prepare the content to analyze
    content_to_analyze = ""
    if isinstance(scraped_data, dict):
        # Log what we received for debugging
        print("\n" + "="*80)
        print("📋 SUMMARIZER - Received Data Structure")
        print("="*80)
        print(f"Job Title: {scraped_data.get('job_title', 'Not provided')}")
        print(f"Company Name (pre-extracted): {scraped_data.get('company_name', 'Not provided')}")
        print(f"Location: {scraped_data.get('location', 'Not provided')}")
        print(f"Description length: {len(str(scraped_data.get('description', '')))} chars")
        print(f"Text content length: {len(str(scraped_data.get('text_content', '')))} chars")
        print(f"Description preview (first 200 chars): {str(scraped_data.get('description', ''))[:200]}...")
        print("="*80 + "\n")
        
        # Combine all relevant fields
        content_parts = []
        
        if scraped_data.get("text_content"):
            content_parts.append(f"Full Page Text:\n{scraped_data['text_content']}")
        
        if scraped_data.get("description"):
            content_parts.append(f"Description:\n{scraped_data['description']}")
        
        if scraped_data.get("qualifications"):
            content_parts.append(f"Qualifications:\n{scraped_data['qualifications']}")
        
        if scraped_data.get("suggested_skills"):
            content_parts.append(f"Suggested Skills:\n{scraped_data['suggested_skills']}")
        
        if scraped_data.get("job_title"):
            content_parts.append(f"Job Title: {scraped_data['job_title']}")
        
        if scraped_data.get("company_name"):
            content_parts.append(f"Company: {scraped_data['company_name']}")
        
        if scraped_data.get("location"):
            content_parts.append(f"Location: {scraped_data['location']}")
        
        content_to_analyze = "\n\n".join(content_parts) if content_parts else str(scraped_data)
    else:
        content_to_analyze = str(scraped_data)
        print(f"[SUMMARIZER] Received non-dict data: {type(scraped_data)}")
    
    # Create extraction prompt (same structure as app.py lines 1728-1742)
    extraction_prompt = f"""Given the following scraped job posting data, extract ALL available information and return it in a structured format.

Extract these fields:
- Job title (exact title from posting)
- Company name
- Complete job description
- Required skills (list each skill separately)
- Required experience (years and type)
- Qualifications and education requirements
- Responsibilities
- Salary/compensation (if mentioned)
- Location
- Job type (full-time, internship, etc.)
- Visa sponsorship or scholarship information (if mentioned - look for keywords like: visa sponsorship, visa support, H1B, work permit, scholarship, funding, financial support, tuition assistance, etc.)

Return structured data with all fields clearly labeled.
If a field is not found, mark it as 'Not specified'.
For visa/scholarship information: If mentioned, extract the exact details. If not mentioned, set to 'Not specified'.

Content:
{content_to_analyze}"""
    
    try:
        # Run agent
        agent_response = agent.run(extraction_prompt)
        
        # Extract response content
        response_text = ""
        if hasattr(agent_response, 'content'):
            response_text = str(agent_response.content)
        elif hasattr(agent_response, 'messages') and agent_response.messages:
            last_msg = agent_response.messages[-1]
            response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
        else:
            response_text = str(agent_response)
        
        # Parse the structured response
        structured_data = _parse_agent_response(response_text, scraped_data)
        
        return structured_data
        
    except Exception as e:
        # Fallback: return basic structure from scraped_data
        print(f"[RESPONSE] Error in agent summarization: {e}")
        return _create_fallback_response(scraped_data)


def _parse_agent_response(response_text: str, scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse agent response to extract structured fields.
    """
    result = _empty_result()

    # Try to parse JSON if agent responded with structured JSON
    json_payload = _extract_json_payload(response_text)
    if json_payload:
        return _result_from_json(json_payload, scraped_data, result)

    # Use regex to extract fields from agent response
    patterns = {
        "job_title": r'(?:Job Title|Title)[:\s]+(.+?)(?:\n|$)',
        "company_name": r'(?:Company Name|Company)[:\s]+(.+?)(?:\n|$)',
        "location": r'(?:Location)[:\s]+(.+?)(?:\n|$)',
        "required_experience": r'(?:Required Experience|Experience|Years of Experience)[:\s]+(.+?)(?:\n|$)',
        "salary": r'(?:Salary|Compensation|Pay)[:\s]+(.+?)(?:\n|$)',
        "job_type": r'(?:Job Type|Type|Employment Type)[:\s]+(.+?)(?:\n|$)',
        "visa_scholarship_info": r'(?:Visa Sponsorship|Visa Support|Scholarship|Visa/Scholarship|Visa and Scholarship)[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            result[field] = match.group(1).strip()
    
    # Extract description (everything after "Description:" or "Job Description:")
    desc_match = re.search(
        r'(?:Description|Job Description|Complete Job Description)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Responsibilities)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if desc_match:
        result["description"] = desc_match.group(1).strip()
    
    # Extract qualifications
    qual_match = re.search(
        r'(?:Qualifications|Education Requirements|Qualifications and Education)[:\s]+(.+?)(?:\n\n(?:Required|Responsibilities|Salary)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if qual_match:
        result["qualifications"] = qual_match.group(1).strip()
    
    # Extract responsibilities
    resp_match = re.search(
        r'(?:Responsibilities|Core Responsibilities|Duties)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Salary)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if resp_match:
        result["responsibilities"] = resp_match.group(1).strip()
    
    # Extract required skills (list)
    skills_section = re.search(
        r'(?:Required Skills|Skills Needed|Skills Required)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Responsibilities|Experience)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if skills_section:
        skills_text = skills_section.group(1).strip()
        # Split by newlines, bullets, or commas
        result["required_skills"] = [
            skill.strip() for skill in re.split(r'[\n•,\-]', skills_text)
            if skill.strip() and skill.strip() != "Not specified"
        ]
    
    # Fallback to scraped_data if fields are missing
    if not result["job_title"] and scraped_data.get("job_title"):
        result["job_title"] = scraped_data["job_title"]
    
    if not result["company_name"] and scraped_data.get("company_name"):
        result["company_name"] = scraped_data["company_name"]
    
    if not result["location"] and scraped_data.get("location"):
        result["location"] = scraped_data["location"]
    
    if not result["description"] and scraped_data.get("description"):
        result["description"] = scraped_data["description"]
    
    if not result["qualifications"] and scraped_data.get("qualifications"):
        result["qualifications"] = scraped_data["qualifications"]
    
    if not result["suggested_skills"] and scraped_data.get("suggested_skills"):
        # Parse suggested skills from scraped data
        skills_text = scraped_data["suggested_skills"]
        result["suggested_skills"] = [
            skill.strip() for skill in re.split(r'[\n•,\-]', skills_text)
            if skill.strip()
        ]
    
    # If description is still None, use full response as description
    if not result["description"]:
        result["description"] = response_text.strip()
    
    # Extract visa/scholarship info with broader search if not found
    if not result["visa_scholarship_info"] or result["visa_scholarship_info"] == "Not specified":
        # Look for visa/scholarship keywords in the full response
        visa_keywords = [
            r'visa sponsorship[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'visa support[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'scholarship[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'H1B[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'work permit[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'financial support[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'tuition assistance[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
        ]
        for pattern in visa_keywords:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                result["visa_scholarship_info"] = match.group(1).strip()
                break
        
        # Also check in scraped_data text_content
        if (not result["visa_scholarship_info"] or result["visa_scholarship_info"] == "Not specified") and scraped_data.get("text_content"):
            text_content = scraped_data["text_content"].lower()
            if any(keyword in text_content for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit", "financial support", "tuition"]):
                # Extract surrounding context
                for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit"]:
                    if keyword in text_content:
                        # Find the sentence or paragraph containing the keyword
                        idx = text_content.find(keyword)
                        start = max(0, idx - 100)
                        end = min(len(text_content), idx + 200)
                        context = scraped_data["text_content"][start:end].strip()
                        result["visa_scholarship_info"] = context
                        break
            else:
                result["visa_scholarship_info"] = "Not specified"
        elif not result["visa_scholarship_info"]:
            result["visa_scholarship_info"] = "Not specified"
    
    return _finalize_result(result, scraped_data)


def _create_fallback_response(scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a fallback response structure from scraped_data if agent fails.
    """
    # Check for visa/scholarship info in scraped data
    visa_scholarship_info = "Not specified"
    text_content = scraped_data.get("text_content", "").lower() if scraped_data.get("text_content") else ""
    if any(keyword in text_content for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit", "financial support", "tuition"]):
        # Extract context around visa/scholarship keywords
        for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit"]:
            if keyword in text_content:
                idx = text_content.find(keyword)
                start = max(0, idx - 100)
                end = min(len(text_content), idx + 200)
                context = scraped_data.get("text_content", "")[start:end].strip()
                visa_scholarship_info = context
                break
    
    return {
        "job_title": scraped_data.get("job_title"),
        "company_name": scraped_data.get("company_name"),
        "location": scraped_data.get("location"),
        "description": scraped_data.get("description") or scraped_data.get("text_content", "")[:2000],
        "required_skills": [],
        "required_experience": None,
        "qualifications": scraped_data.get("qualifications"),
        "responsibilities": None,
        "salary": None,
        "job_type": None,
        "suggested_skills": scraped_data.get("suggested_skills", "").split("\n") if scraped_data.get("suggested_skills") else [],
        "visa_scholarship_info": visa_scholarship_info
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "job_title": None,
        "company_name": None,
        "location": None,
        "description": None,
        "required_skills": [],
        "required_experience": None,
        "qualifications": None,
        "responsibilities": None,
        "salary": None,
        "job_type": None,
        "suggested_skills": [],
        "visa_scholarship_info": None
    }


def _extract_json_payload(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON payload from response text if present."""
    text = response_text.strip()

    # Remove leading/trailing backticks or code fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Look for first JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return None

    json_candidate = json_match.group(0)

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        # Try to clean up trailing commas or quotes
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _result_from_json(
    payload: Dict[str, Any],
    scraped_data: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Populate result from JSON payload."""
    field_mapping = {
        "job_title": ["job_title", "Job title", "Title"],
        "company_name": ["company_name", "Company name", "Company"],
        "location": ["location", "Location"],
        "description": ["description", "Complete job description", "Job description"],
        "required_skills": ["required_skills", "Required skills"],
        "required_experience": ["required_experience", "Required experience", "Experience"],
        "qualifications": ["qualifications", "Qualifications and education requirements", "Qualifications"],
        "responsibilities": ["responsibilities", "Responsibilities"],
        "salary": ["salary", "Salary/compensation", "Compensation"],
        "job_type": ["job_type", "Job type", "Type"],
        "suggested_skills": ["suggested_skills", "Suggested skills"],
        "visa_scholarship_info": ["visa_scholarship_info", "Visa sponsorship or scholarship information"],
    }

    normalized_payload = {str(k).strip(): v for k, v in payload.items()}

    for field, keys in field_mapping.items():
        for key in keys:
            if key in normalized_payload and normalized_payload[key] not in (None, ""):
                value = normalized_payload[key]
                if isinstance(value, str):
                    value = value.strip().strip('"')
                    if value.lower() == "not specified":
                        value = "Not specified"
                if field in {"required_skills", "suggested_skills"} and isinstance(value, str):
                    value = _split_to_list(value)
                result[field] = value
                break

    return _finalize_result(result, scraped_data)


def _split_to_list(value: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"[\n•,\-]", value)
        if item.strip() and item.strip().lower() != "not specified"
    ]


def _finalize_result(result: Dict[str, Any], scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply fallbacks and normalization before returning result."""
    # Fallback to scraped_data if fields are missing
    if not result["job_title"] and scraped_data.get("job_title"):
        result["job_title"] = scraped_data["job_title"]

    if not result["company_name"] and scraped_data.get("company_name"):
        result["company_name"] = scraped_data["company_name"]

    if not result["location"] and scraped_data.get("location"):
        result["location"] = scraped_data["location"]

    if not result["description"] and scraped_data.get("description"):
        result["description"] = scraped_data["description"]

    if not result["qualifications"] and scraped_data.get("qualifications"):
        result["qualifications"] = scraped_data["qualifications"]

    if not result["suggested_skills"] and scraped_data.get("suggested_skills"):
        result["suggested_skills"] = _split_to_list(scraped_data["suggested_skills"])

    if isinstance(result["required_skills"], str):
        result["required_skills"] = _split_to_list(result["required_skills"])

    if isinstance(result["suggested_skills"], str):
        result["suggested_skills"] = _split_to_list(result["suggested_skills"])

    if not result["description"]:
        result["description"] = "Not specified"

    if not result["visa_scholarship_info"]:
        result["visa_scholarship_info"] = "Not specified"
    else:
        result["visa_scholarship_info"] = result["visa_scholarship_info"].strip()

    if isinstance(result["company_name"], str):
        result["company_name"] = re.sub(r'^\*+\s*|\s*\*+$', '', result["company_name"]).strip()
        result["company_name"] = result["company_name"].replace('",', '').strip()
        if result["company_name"].lower() == "not specified":
            result["company_name"] = "Not specified"

    if "is_authorized_sponsor" not in result or result["is_authorized_sponsor"] is None:
        result["is_authorized_sponsor"] = scraped_data.get("is_authorized_sponsor")

    return result


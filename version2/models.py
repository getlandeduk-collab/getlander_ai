from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator, ConfigDict


class ResumeInfo(BaseModel):
    name: str
    size: str
    type: str
    content: Optional[str] = None  # Base64 encoded PDF content


class JobInfo(BaseModel):
    title: str
    company: str
    url: HttpUrl


class MatchJobsRequest(BaseModel):
    resume: ResumeInfo
    jobs: List[JobInfo] = Field(default_factory=list)
    user_id: Optional[str] = Field(
        default=None,
        description="User ID to save job applications to Firestore"
    )

    @validator("jobs")
    def validate_jobs(cls, v: List[JobInfo]) -> List[JobInfo]:
        if len(v) == 0:
            raise ValueError("At least one job is required")
        if len(v) > 40:
            raise ValueError("A maximum of 40 jobs is allowed")
        return v


class MatchJobsJsonRequest(BaseModel):
    pdf: Optional[str] = Field(
        default=None,
        description="Base64-encoded PDF content of the resume",
    )
    urls: List[HttpUrl] = Field(
        default_factory=list,
        description="List of job posting URLs to analyze (max 40)",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID to save job applications to Firestore"
    )

    @validator("urls")
    def validate_urls(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("At least one URL is required")
        if len(v) > 40:
            raise ValueError("A maximum of 40 URLs is allowed")
        return v


class CandidateProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_summary: Optional[str] = None
    total_years_experience: Optional[float] = None
    interests: List[str] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text_excerpt: Optional[str] = None


class JobPosting(BaseModel):
    url: HttpUrl
    job_title: Optional[str] = None
    company: Optional[str] = None
    requirements: List[str] = Field(default_factory=list)
    skills_needed: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None
    description: Optional[str] = None
    salary: Optional[str] = None


class MatchedJob(BaseModel):
    rank: int
    job_url: HttpUrl
    job_title: Optional[str]
    company: Optional[str]
    match_score: float = Field(ge=0.0, le=1.0)
    summary: Optional[str]
    key_matches: List[str] = Field(default_factory=list)
    requirements_met: int = 0
    total_requirements: int = 0
    scraped_summary: Optional[str] = None  # Brief summary of scraped content
    visa_scholarship_info: Optional[str] = Field(
        default=None,
        description="Information about visa sponsorship or scholarship opportunities (if mentioned in job posting)"
    )


class SponsorshipInfo(BaseModel):
    """Sponsorship information for a company."""
    company_name: Optional[str] = None
    sponsors_workers: bool = False
    visa_types: Optional[str] = None
    summary: str = "No sponsorship information available"


class MatchJobsResponse(BaseModel):
    candidate_profile: CandidateProfile
    matched_jobs: List[MatchedJob]
    processing_time: str
    jobs_analyzed: int
    request_id: str
    sponsorship: Optional[SponsorshipInfo] = None


class ProgressStatus(BaseModel):
    request_id: str
    status: str = Field(description="queued|parsing|scraping|matching|summarizing|completed|error")
    message: Optional[str] = None
    jobs_total: int = 0
    jobs_scraped: int = 0
    jobs_cached: int = 0
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    openai_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    model_name: str = "gpt-4o"
    request_timeout_seconds: int = 120
    max_concurrent_scrapes: int = 8
    rate_limit_requests_per_minute: int = 60
    cache_ttl_seconds: int = 3600


class FirebaseResume(BaseModel):
    """Model for a Firebase resume document."""
    id: str
    name: Optional[str] = None
    size: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None  # Base64 encoded PDF content (may have PDF_BASE64: prefix)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class FirebaseResumeListResponse(BaseModel):
    """Response model for listing user resumes."""
    user_id: str
    resumes: List[FirebaseResume]
    count: int


class FirebaseResumeResponse(BaseModel):
    """Response model for a single resume."""
    user_id: str
    resume: FirebaseResume


class SavedCVResponse(BaseModel):
    """Response model for savedCVs array."""
    user_id: str
    saved_cvs: List[Dict[str, Any]]
    count: int


# Request models for Firebase endpoints (POST requests)
class GetUserResumesRequest(BaseModel):
    """Request model for getting all resumes for a user."""
    user_id: str = Field(..., description="The user ID to fetch resumes for")


class GetUserResumeRequest(BaseModel):
    """Request model for getting a specific resume."""
    user_id: str = Field(..., description="The user ID")
    resume_id: str = Field(..., description="The resume document ID")


class GetUserResumePdfRequest(BaseModel):
    """Request model for getting a resume PDF."""
    user_id: str = Field(..., description="The user ID")
    resume_id: str = Field(..., description="The resume document ID")


class GetUserResumeBase64Request(BaseModel):
    """Request model for getting a resume as base64."""
    user_id: str = Field(..., description="The user ID")
    resume_id: str = Field(..., description="The resume document ID")


class GetUserSavedCvsRequest(BaseModel):
    """Request model for getting savedCVs array."""
    user_id: str = Field(..., description="The user ID")


# Job extraction models
class ExtractJobInfoRequest(BaseModel):
    """Request model for extracting job information from a URL."""
    job_url: HttpUrl = Field(..., description="The job posting URL to extract information from")


class JobInfoExtracted(BaseModel):
    """Response model for extracted job information."""
    job_url: str
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    portal: Optional[str] = None  # e.g., "LinkedIn", "Internshala", "Indeed", etc.
    description: Optional[str] = None  # Generated by scraper and summarizer agents
    visa_scholarship_info: Optional[str] = Field(
        default=None,
        description="Information about visa sponsorship or scholarship opportunities (if mentioned in job posting)"
    )
    success: bool = True
    error: Optional[str] = None


# Playwright scraper models
class PlaywrightScrapeRequest(BaseModel):
    """Request model for playwright scraper endpoint."""
    url: HttpUrl = Field(..., description="The job posting URL to scrape")
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for agent summarization (optional, uses env var if not provided)"
    )


class PlaywrightScrapeResponse(BaseModel):
    """Response model for playwright scraper with agent summarization."""
    url: str
    scraped_data: Dict[str, Any] = Field(..., description="Raw scraped data from playwright")
    summarized_data: Dict[str, Any] = Field(..., description="Structured data from agent summarization")
    portal: Optional[str] = Field(
        default=None,
        description="Detected job portal (e.g., LinkedIn, Indeed, etc.)"
    )
    is_authorized_sponsor: Optional[bool] = Field(
        default=None,
        description="Whether the company appears on the authorized work visa sponsor list"
    )
    match_score: Optional[float] = Field(default=None, description="Job-candidate match score (0.0-1.0)")
    key_matches: Optional[List[str]] = Field(default=None, description="Key matching qualifications")
    requirements_met: Optional[int] = Field(default=None, description="Number of requirements met")
    total_requirements: Optional[int] = Field(default=None, description="Total number of requirements")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the match score")
    visa_scholarship_info: Optional[str] = Field(
        default=None,
        description="Information about visa sponsorship or scholarship opportunities (if mentioned in job posting)"
    )
    success: bool = True
    error: Optional[str] = None


# Summarizer-only endpoint models
class SummarizeJobRequest(BaseModel):
    """Request model for summarizer endpoint - accepts preprocessed scraped data."""
    scraped_data: Dict[str, Any] = Field(
        ...,
        description="Preprocessed scraped data from backend (structured data)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for agent summarization (optional, uses env var if not provided)"
    )


class SummarizeJobResponse(BaseModel):
    """Response model for summarizer endpoint."""
    summarized_data: Dict[str, Any] = Field(
        ...,
        description="Structured data from agent summarization"
    )
    success: bool = True
    error: Optional[str] = None



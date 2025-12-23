# Deterministic Job-Resume Matching System

A two-step matching system that combines LLM extraction with deterministic mathematical scoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT                                     │
│  Job Description (text) + Resume (text)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: LLM EXTRACTION                          │
│  (LangGraph + GPT-4, temperature=0)                           │
│                                                              │
│  Extracts structured data:                                  │
│  - Job: skills_required, skills_preferred, years_experience │
│  - Resume: candidate_skills, candidate_years, etc.          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         STEP 2: DETERMINISTIC SCORING                        │
│  (Pure Python, no AI)                                       │
│                                                              │
│  Component Scores (weighted):                               │
│  1. Skills Match (35%)                                      │
│  2. Experience Match (25%)                                  │
│  3. Responsibility Match (20%)                              │
│  4. Education Match (10%)                                   │
│  5. Seniority Match (10%)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT                                     │
│  {                                                           │
│    "match_percentage": 72.5,                                │
│    "breakdown": {...},                                      │
│    "extracted_data": {...}                                  │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd version2
pip install -r requirements.txt  # Ensure scikit-learn, phi, openai are installed
```

## Quick Start

```python
from matching import match_job_resume

# Match a job and resume
result = match_job_resume(job_description, resume_text)

print(f"Match: {result['match_percentage']}%")
print(f"Skills: {result['breakdown']['skills']}%")
print(f"Experience: {result['breakdown']['experience']}%")
```

## Usage Examples

### Basic Matching

```python
from matching import match_job_resume

job_desc = """
Senior Python Developer
Requirements:
- 3+ years Python experience
- Django, Flask
- SQL databases
- Bachelor's in CS
"""

resume = """
John Doe
BS Computer Science, 2019
5 years Python development
Skills: Python, Django, PostgreSQL, AWS
"""

result = match_job_resume(job_desc, resume)
# Result: ~85% match
```

### Multiple Jobs

```python
from matching.matcher import match_multiple_jobs

jobs = [job1_desc, job2_desc, job3_desc]
results = match_multiple_jobs(jobs, resume_text)

for i, result in enumerate(results, 1):
    print(f"#{i}: {result['match_percentage']}%")
```

### Custom Model

```python
result = match_job_resume(
    job_desc,
    resume,
    model_name="gpt-4o-mini"  # Use different model
)
```

## Configuration

Edit `matching/config.py` to adjust:

- **Weights**: Change component importance
- **Penalties**: Adjust experience penalty
- **Field Groups**: Add related education fields
- **Skill Normalization**: Add skill mappings

```python
# config.py
WEIGHTS = {
    "skills": 0.40,  # Increase skills weight
    "experience": 0.20,  # Decrease experience weight
    ...
}
```

## Scoring Methodology

### 1. Skills Match (35%)

```
Required Score = (matched_required / total_required) * 100
Preferred Score = (matched_preferred / total_preferred) * 100
Final = (0.70 * Required) + (0.30 * Preferred)
```

### 2. Experience Match (25%)

```
If candidate >= required:
    Score = min(100, (candidate / required) * 100)
Else:
    Score = (candidate / required) * 100 * 0.7  # Penalty
```

### 3. Responsibility Match (20%)

Uses TF-IDF + Cosine Similarity (or Jaccard if sklearn unavailable)

### 4. Education Match (10%)

```
Level Score = Based on degree hierarchy
Field Multiplier = 1.0 (exact), 0.85 (related), 0.6 (different)
Final = Level Score * Field Multiplier
```

### 5. Seniority Match (10%)

```
0 levels difference: 100
1 level above: 90
1 level below: 60
2+ levels: 30
```

## Testing

```bash
# Run all tests
python -m unittest matching/tests/test_matcher.py

# Run specific test
python -m unittest matching.tests.test_matcher.TestScoringComponents.test_skills_score_perfect_match

# With verbose output
python -m unittest matching/tests/test_matcher.py -v
```

## API Reference

### `match_job_resume(job_description, resume, model_name=None, return_extracted_data=True)`

Main matching function.

**Parameters:**
- `job_description` (str): Full job description text
- `resume` (str): Full resume text
- `model_name` (str, optional): LLM model name
- `return_extracted_data` (bool): Include extracted data in response

**Returns:**
```python
{
    "match_percentage": float,  # 0-100
    "breakdown": {
        "skills": float,
        "experience": float,
        "responsibilities": float,
        "education": float,
        "seniority": float
    },
    "extracted_data": {  # Optional
        "job": {...},
        "resume": {...}
    }
}
```

### `match_multiple_jobs(job_descriptions, resume, model_name=None)`

Match resume against multiple jobs.

**Returns:** List of match results, sorted by match_percentage (highest first)

## File Structure

```
matching/
├── __init__.py           # Package exports
├── config.py             # Configuration & weights
├── llm_extractor.py      # LLM extraction logic
├── scoring_engine.py     # Deterministic scoring
├── matcher.py            # Main orchestration
├── README.md             # This file
└── tests/
    ├── __init__.py
    └── test_matcher.py   # Unit tests
```

## Key Features

✅ **Deterministic**: Same inputs = same outputs (for scoring)
✅ **Explainable**: Detailed breakdown of each component
✅ **Configurable**: Easy to adjust weights and parameters
✅ **Tested**: Comprehensive unit tests
✅ **Modular**: Clean separation of concerns
✅ **Consistent**: LLM uses temperature=0 for extraction

## Troubleshooting

### "OPENAI_API_KEY not set"
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "sklearn not available"
Install scikit-learn:
```bash
pip install scikit-learn
```

### Low match scores
- Check if skills are normalized correctly
- Adjust weights in `config.py`
- Review extracted data to ensure LLM extracted correctly

### Inconsistent extraction
- Ensure temperature=0 in config
- Check LLM model supports temperature
- Review extraction prompt in `llm_extractor.py`

## Examples

See `tests/test_matcher.py` for complete examples with sample data.

## License

Part of the CHROME_EXT_SHAMEER project.


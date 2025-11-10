from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import requests
from bs4 import BeautifulSoup
import json
import re
import os
import sys
import argparse
import logging
from urllib.parse import urlparse, urljoin
from Scraper import scrape_internshala


class InternshipRequest(BaseModel):
    url: HttpUrl
    categories: Optional[List[str]] = None
    role: Optional[str] = Field(default=None, alias="Role")
    types: Optional[List[str]] = None
    instruction: Optional[str] = None
    resume: Optional[str] = None
    force_selenium: Optional[bool] = Field(default=False, description="Force Selenium rendering if True")

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True


def parse_json_ld_jobs(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(script.string or "{}")
        except json.JSONDecodeError:
            continue

        def normalize_to_list(obj: Any) -> List[Any]:
            if obj is None:
                return []
            if isinstance(obj, list):
                return obj
            return [obj]

        stack = normalize_to_list(data)
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type")
            if isinstance(node_type, list):
                is_job = any(t == "JobPosting" for t in node_type)
            else:
                is_job = node_type == "JobPosting"
            if is_job:
                jobs.append({
                    "title": (node.get("title") or node.get("jobTitle") or "").strip(),
                    "description": (node.get("description") or "").strip(),
                    "type": (node.get("employmentType") or "").strip(),
                    "company": (node.get("hiringOrganization", {}) or {}).get("name", "") if isinstance(node.get("hiringOrganization"), dict) else (node.get("hiringOrganization") or ""),
                    "location": (node.get("jobLocation", {}) or {}).get("address", {}).get("addressLocality", "") if isinstance(node.get("jobLocation"), dict) else "",
                    "applyUrl": (node.get("hiringOrganization", {}) or {}).get("sameAs") or node.get("url") or "",
                    "category": "",
                    "source": "json-ld"
                })
            # Explore nested structures
            for key in ("@graph", "itemListElement", "hasPart", "mainEntity"):
                if key in node:
                    stack.extend(normalize_to_list(node[key]))

    return jobs


def parse_microdata_jobs(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for tag in soup.find_all(attrs={"itemscope": True, "itemtype": True}):
        itemtype = tag.get("itemtype", "")
        if "JobPosting" not in itemtype:
            continue
        def get_itemprop(prop: str) -> str:
            el = tag.find(attrs={"itemprop": prop})
            if not el:
                return ""
            if el.name == "meta":
                return (el.get("content") or "").strip()
            return (el.get_text(" ", strip=True) or "").strip()
        jobs.append({
            "title": get_itemprop("title") or get_itemprop("jobTitle"),
            "description": get_itemprop("description"),
            "type": get_itemprop("employmentType"),
            "company": get_itemprop("hiringOrganization"),
            "location": get_itemprop("jobLocation") or get_itemprop("addressLocality"),
            "applyUrl": get_itemprop("url"),
            "category": "",
            "source": "microdata"
        })
    return jobs


JOB_TYPE_PATTERN = re.compile(r"\b(full[- ]?time|part[- ]?time|contract|intern(ship)?|temporary|freelance|remote)\b", re.I)


def heuristic_jobs(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    candidate_containers = []
    for tag in soup.find_all(["section", "div", "article", "li"]):
        classes = " ".join(tag.get("class", []))
        ident = tag.get("id", "")
        label = f"{classes} {ident}".lower()
        if any(k in label for k in ["job", "position", "opening", "career", "vacancy", "opportunity", "intern", "internship"]):
            candidate_containers.append(tag)

    for container in candidate_containers:
        title_el = None
        for name in ["h1", "h2", "h3", "h4"]:
            title_el = container.find(name)
            if title_el:
                break
        if not title_el:
            title_el = container.find("a")
        title = (title_el.get_text(" ", strip=True) if title_el else "").strip()
        if not title:
            continue
        text = container.get_text(" ", strip=True)
        type_match = JOB_TYPE_PATTERN.search(text or "")
        job_type = type_match.group(0) if type_match else ""
        apply_link = ""
        for a in container.find_all("a", href=True):
            href = a.get("href") or ""
            if any(k in href.lower() for k in ["apply", "jobs", "careers", "position", "opening", "intern"]):
                apply_link = href
                break
        jobs.append({
            "title": title,
            "description": text[:2000],
            "type": job_type,
            "company": "",
            "location": "",
            "applyUrl": apply_link,
            "category": "",
            "source": "heuristic"
        })
    return jobs


def extract_internshala_jobs(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    # Common containers on internshala
    containers = soup.select('div.individual_internship, div.internship_meta, div.container-fluid.individual_internship')
    if not containers:
        # Fallback: cards with data attributes
        containers = [c for c in soup.find_all('div') if 'intern' in ' '.join(c.get('class', [])).lower()]
    for c in containers:
        # Title/profile
        title = ""
        title_el = c.select_one('div.profile') or c.select_one('h3') or c.find('a')
        if title_el:
            title = title_el.get_text(" ", strip=True)
        # Company
        company = ""
        company_el = c.select_one('a.link_display_like_text') or c.select_one('div.company_name')
        if company_el:
            company = company_el.get_text(" ", strip=True)
        # Location
        location = ""
        loc_el = c.select_one('a.location_link') or c.find('span', string=re.compile(r'Location', re.I))
        if loc_el:
            location = loc_el.get_text(" ", strip=True)
        # Type (internship / full-time etc.)
        job_type = ""
        type_el = c.find(string=re.compile(r'internship|full[- ]?time|part[- ]?time', re.I))
        if type_el:
            job_type = type_el.strip()
        # Apply URL
        apply_url = ""
        apply_link_el = c.select_one('a.view_detail_button, a[href*="internship/details"], a[href*="/internship/"]') or c.find('a', href=True)
        if apply_link_el and apply_link_el.get('href'):
            apply_url = urljoin(base_url, apply_link_el.get('href'))
        # Description snippet
        desc = ""
        desc_el = c.select_one('div.description, div.job-description') or c
        if desc_el:
            desc = desc_el.get_text(" ", strip=True)[:2000]
        if title or desc:
            jobs.append({
                "title": title,
                "description": desc,
                "type": job_type,
                "company": company,
                "location": location,
                "applyUrl": apply_url,
                "category": "",
                "source": "internshala"
            })
    return jobs


def merge_and_deduplicate(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    def make_key(j: Dict[str, Any]) -> str:
        title = (j.get("title") or "").lower().strip()
        company = (j.get("company") or "").lower().strip()
        return f"{title}|{company}"
    for j in jobs:
        key = make_key(j)
        if not key.strip("|"):
            key = (j.get("description") or "")[:60]
        if key not in seen:
            seen[key] = j
        else:
            priority = {"internshala": 3, "json-ld": 2, "microdata": 1, "heuristic": 0}
            if priority.get(j.get("source"), 0) > priority.get(seen[key].get("source"), 0):
                seen[key] = j
    return list(seen.values())


def filter_jobs(jobs: List[Dict[str, Any]], categories: Optional[List[str]], types: Optional[List[str]], role: Optional[str]) -> List[Dict[str, Any]]:
    def matches(job: Dict[str, Any]) -> bool:
        if categories:
            cat_text = (job.get("category") or job.get("description") or job.get("title") or "").lower()
            if not any(c.lower() in cat_text for c in categories):
                return False
        if types:
            job_type = (job.get("type") or "").lower()
            if not any(t.lower() in job_type for t in types):
                return False
        if role:
            blob = " ".join([
                job.get("title") or "",
                job.get("description") or "",
            ]).lower()
            if role.lower() not in blob:
                return False
        return True
    return [j for j in jobs if matches(j)]


def get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def discover_pagination_links(soup: BeautifulSoup, base_url: str, max_links: int = 5) -> List[str]:
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text(" ", strip=True) or "").lower()
        href = a.get("href") or ""
        if any(k in text for k in ["next", ">", "more"]) or re.search(r"page|pagination|p=\d+", href, re.I):
            full = urljoin(base_url, href)
            if full not in links:
                links.append(full)
        if len(links) >= max_links:
            break
    return links


def try_retriever_site_search(query: str, site_domain: str) -> List[str]:
    candidates: List[str] = []
    search_query = f"site:{site_domain} {query} internships jobs career opening"
    try:
        from retrievers.google.google import GoogleSearch
        r = GoogleSearch(search_query)
        results = r.search(max_results=7) or []
        candidates.extend([item.get("href") for item in results if item.get("href")])
        return candidates
    except Exception:
        pass
    try:
        from retrievers.tavily.tavily_search import TavilySearch
        r = TavilySearch(search_query)
        results = r.search(max_results=7) or []
        candidates.extend([item.get("href") for item in results if item.get("href")])
        return candidates
    except Exception:
        pass
    try:
        from retrievers.duckduckgo.duckduckgo import Duckduckgo
        r = Duckduckgo(search_query)
        results = r.search(max_results=7) or []
        candidates.extend([item.get("href") for item in results if item.get("href")])
        return candidates
    except Exception:
        pass
    return candidates


def build_internship_prompt(jobs: List[Dict[str, Any]], instruction: Optional[str]) -> List[Dict[str, str]]:
    system_text = "You are an assistant that recommends internships based on a user's resume and preferences."
    user_intro = instruction or "Summarize and recommend the best matching internships."

    def shorten(text: str, limit: int = 1000) -> str:
        return text[:limit] if text else ""

    lines = [user_intro, "", "Internships:"]
    for idx, j in enumerate(jobs, start=1):
        lines.append(f"{idx}. Title: {j.get('title','')}")
        lines.append(f"   Type: {j.get('type','')} | Location: {j.get('location','')}")
        lines.append(f"   Apply: {j.get('applyUrl','')}")
        lines.append(f"   Description: {shorten(j.get('description',''))}")
        lines.append("")

    user_text = "\n".join(lines)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


app = FastAPI(title="Internships API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/internships")
def get_internships(payload: InternshipRequest) -> Dict[str, Any]:
    logger = logging.getLogger("api")
    logger.info("/internships: start url=%s role=%s cats=%s types=%s force_selenium=%s", str(payload.url), payload.role, payload.categories, payload.types, payload.force_selenium)
    # For Internshala, use our focused scraper for robust results
    listings_raw = scrape_internshala(str(payload.url), max_pages=2, use_selenium_if_needed=bool(payload.force_selenium))
    # Normalize keys to a consistent schema
    listings: List[Dict[str, Any]] = []
    for it in listings_raw:
        listings.append({
            "title": it.get("title", ""),
            "company": it.get("company", ""),
            "location": it.get("location", ""),
            "duration": it.get("duration", ""),
            "stipend": it.get("stipend", ""),
            "apply_link": it.get("apply_link", ""),
            "description": it.get("description", ""),
            "type": "internship",
            "category": "",
        })

    logger.info("/internships: scraped count=%d", len(listings))
    filtered = filter_jobs(
        [
            {
                **j,
                # Back-compat for filter_jobs which expects certain keys
                "applyUrl": j.get("apply_link", ""),
            }
            for j in listings
        ],
        payload.categories, payload.types, payload.role,
    )
    # Drop back-compat key
    filtered = [
        {k: v for k, v in j.items() if k != "applyUrl"}
        for j in filtered
    ]
    logger.info("/internships: filtered count=%d", len(filtered))

    result: Dict[str, Any] = {"count": len(filtered), "internships": filtered}

    # Optional LLM recommendation step
    if payload.instruction and filtered:
        try:
            from llm_provider.generic.base import GenericLLMProvider
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            llm = GenericLLMProvider.from_provider(provider="openai", model=model_name, temperature=0.2)
            messages = build_internship_prompt(filtered, payload.instruction)
            import asyncio
            async def _run():
                return await llm.get_chat_response(messages, stream=False)
            summary = asyncio.run(_run())
            if summary:
                result["summary"] = summary
        except Exception as e:
            logger.exception("/internships: llm_summary_failed err=%s", e)

    # Semantic matching if resume provided
    if payload.resume and filtered:
        logger.info("/internships: matching resume length=%d", len(payload.resume or ""))
        try:
            result["internships_matched"] = match_jobs_to_resume(filtered, payload.resume)
        except Exception as e:
            logger.exception("/internships: matching_failed err=%s", e)

    return result


# For local development: `uvicorn main:app --reload`


def _safe_import_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception:
        return None


def _cosine_similarity_matrix(a, b):
    import numpy as np
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def match_jobs_to_resume(job_listings: List[Dict[str, Any]], user_resume: str) -> List[Dict[str, Any]]:
    """Compute semantic similarity between resume and each job and return jobs with match_score (0-100)."""
    texts: List[str] = []
    for j in job_listings:
        parts = [
            j.get("title", ""),
            j.get("company", ""),
            j.get("location", ""),
            j.get("duration", ""),
            j.get("stipend", ""),
            j.get("description", ""),
        ]
        texts.append(" \n".join([p for p in parts if p]))

    model = _safe_import_sentence_model()
    scores: List[float] = []
    if model is not None:
        try:
            import numpy as np
            job_emb = model.encode(texts, normalize_embeddings=True)
            res_emb = model.encode([user_resume], normalize_embeddings=True)
            sims = (job_emb @ res_emb.T).reshape(-1)
            scores = sims.tolist()
        except Exception:
            scores = []

    if not scores:
        # TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            mat = vect.fit_transform(texts + [user_resume])
            sims = cosine_similarity(mat[:-1], mat[-1]).reshape(-1)
            scores = sims.tolist()
        except Exception:
            scores = [0.0] * len(job_listings)

    # Normalize to 0-100
    result: List[Dict[str, Any]] = []
    for j, s in zip(job_listings, scores):
        score_pct = int(round(float(s) * 100))
        result.append({
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "location": j.get("location", ""),
            "duration": j.get("duration", ""),
            "stipend": j.get("stipend", ""),
            "apply_link": j.get("apply_link", j.get("applyUrl", "")),
            "match_score": score_pct,
        })
    # Sort by score desc
    result.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return result


def _print_table(rows: List[Dict[str, Any]]):
    try:
        from tabulate import tabulate
        headers = ["title", "company", "location", "duration", "stipend", "apply_link", "match_score"]
        print(tabulate([[r.get(h, "") for h in headers] for r in rows], headers=headers, tablefmt="github"))
    except Exception:
        print(json.dumps(rows, indent=2, ensure_ascii=False))


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Internshala scraper and matcher")
    parser.add_argument("--input", required=False, help="Path to JSON input or JSON string")
    parser.add_argument("--resume", required=False, help="Path to resume text file or raw resume string")
    parser.add_argument("--as-json", action="store_true", help="Print JSON output instead of table")
    args = parser.parse_args(argv)

    # Default example when no args provided
    if not args.input:
        payload = {
            "url": "https://internshala.com/internships/",
            "categories": ["engineering"],
            "Role": "Machine learning engineer",
            "types": ["full-time"],
            "instruction": "Summarize and group by the {User Resume} and suggest the best internships suitable for them.",
        }
    else:
        # Load from file or parse JSON
        try:
            if os.path.exists(args.input):
                with open(args.input, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = json.loads(args.input)
        except Exception as e:
            print(f"Failed to read input: {e}", file=sys.stderr)
            return 2

    resume_text = ""
    if args.resume:
        if os.path.exists(args.resume):
            with open(args.resume, "r", encoding="utf-8") as f:
                resume_text = f.read()
        else:
            resume_text = args.resume

    req = InternshipRequest.model_validate(payload)
    api_result = get_internships(req)
    internships = api_result.get("internships", [])
    matched = match_jobs_to_resume(internships, resume_text or "") if internships else []

    if args.as_json:
        print(json.dumps(matched, indent=2, ensure_ascii=False))
    else:
        _print_table(matched)
    return 0


if __name__ == "__main__":
    # Basic logging config for CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main_cli())

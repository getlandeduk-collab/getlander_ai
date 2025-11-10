from __future__ import annotations

import base64
import io
import os
import time
import hashlib
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

import pdfplumber

try:
    from firecrawl import FirecrawlApp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FirecrawlApp = None  # type: ignore[assignment]

# Load environment variables to mirror SAMPLE_FIRECRAWL.py behaviour
load_dotenv()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def make_request_id(prefix: str = "req") -> str:
    seed = f"{prefix}-{time.time_ns()}-{os.getpid()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def decode_base64_pdf(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 PDF content") from e


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = 20000) -> str:
    if not pdf_bytes or len(pdf_bytes) < 10:
        raise ValueError("Empty or invalid PDF bytes")
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                txt = ""
            if txt:
                text_parts.append(txt)
            if sum(len(p) for p in text_parts) >= max_chars:
                break
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("Failed to extract text from PDF; consider OCR fallback")
    return text[:max_chars]


def redact_long_text(text: str, max_len: int = 800) -> str:
    if len(text) <= max_len:
        return text
    head = text[: max_len // 2]
    tail = text[-max_len // 2 :]
    return head + "\n...\n" + tail


# Firecrawl-based scraping with compatibility fallbacks
def scrape_website_custom(url: str, api_key: Optional[str]) -> dict:
    if FirecrawlApp is None:
        return {"error": "firecrawl-sdk-missing"}

    firecrawl_api_key = api_key or os.getenv("FIRECRAWL_API_KEY") or ""

    try:
        app = FirecrawlApp(api_key=firecrawl_api_key)
    except Exception as exc:
        return {"error": f"firecrawl-init-failed: {exc}"}

    try:
        result = app.scrape(url=url)
        if isinstance(result, dict):
            return result
        return {"content": str(result)}
    except AttributeError:
        try:
            result = app.scrape_url(url)
            if isinstance(result, dict):
                return result
            return {"content": str(result)}
        except Exception as exc:
            return {"error": f"scrape-failed: {exc}"}
    except Exception as exc:
        return {"error": f"scrape-error: {exc}"}

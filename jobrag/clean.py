"""Turn raw API JSON into one flat, clean list of postings (one schema for all sources)."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from .ingest import is_data_title

GERMAN_HINTS = {"und", "die", "der", "wir", "für", "mit", "sie", "das", "ist", "bei", "eine", "ihre"}
ENGLISH_HINTS = {"and", "the", "we", "for", "with", "you", "our", "are", "is", "your", "will", "to"}


# Footer that Arbeitnow appends to every description; it is not part of the job.
BOILERPLATE_RE = re.compile(r"\s*Find (?:more )?(?:English Speaking )?Jobs in [\w .-]+ on Arbeitnow\s*$", re.I)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")


def html_to_text(raw: str | None) -> str:
    """Strip HTML tags, decode entities and collapse whitespace.

    Some feeds HTML-escape their HTML a second time (``&lt;p&gt;``), so one pass of
    parsing turns entities back into tags. We parse again until no tags are left (max 3).
    """
    if not raw:
        return ""
    text = raw
    for _ in range(3):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = html.unescape(soup.get_text(separator=" "))
        if not _TAG_RE.search(text):
            break
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return BOILERPLATE_RE.sub("", text).strip()


def detect_lang(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())[:400]
    de = sum(w in GERMAN_HINTS for w in words)
    en = sum(w in ENGLISH_HINTS for w in words)
    if de == en == 0:
        return "unknown"
    return "de" if de > en else "en"


_NUM = r"\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?\s?[kK]"
_CUR = r"€|\$|£|eur|usd|gbp|chf"
SALARY_RE = re.compile(
    rf"(?P<c1>{_CUR})?\s?(?P<lo>{_NUM})\s?(?P<c2>{_CUR})?"
    rf"(?:\s?(?:-|–|to|bis)\s?(?P<c3>{_CUR})?\s?(?P<hi>{_NUM})\s?(?P<c4>{_CUR})?)?",
    re.IGNORECASE,
)
CURRENCY = {"€": "EUR", "eur": "EUR", "$": "USD", "usd": "USD", "£": "GBP", "gbp": "GBP", "chf": "CHF"}


def _to_number(s: str) -> float:
    s = s.strip().lower().replace(" ", "")
    if s.endswith("k"):
        return float(s[:-1].replace(",", ".")) * 1000
    return float(re.sub(r"[.,]", "", s))


def parse_salary(text: str | None) -> tuple[float | None, float | None, str | None]:
    """Find the first plausible yearly salary (or range) with a currency in free text.

    Returns (min, max, currency). Only amounts between 10k and 400k with an explicit
    currency are accepted, to avoid picking up phone numbers or headcounts.
    """
    if not text:
        return None, None, None
    for m in SALARY_RE.finditer(text):
        cur = next((m.group(g) for g in ("c1", "c2", "c3", "c4") if m.group(g)), None)
        if not cur:
            continue
        try:
            lo = _to_number(m.group("lo"))
            hi = _to_number(m.group("hi")) if m.group("hi") else lo
        except ValueError:
            continue
        if 10_000 <= lo <= 400_000 and lo <= hi <= 400_000:
            return lo, hi, CURRENCY[cur.lower()]
    return None, None, None


def _ts(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return str(value)


def normalize_remotive(job: dict) -> dict:
    text = html_to_text(job.get("description"))
    smin, smax, cur = parse_salary(job.get("salary") or "")
    return {
        "posting_id": f"remotive-{job['id']}",
        "source": "remotive",
        "title": (job.get("title") or "").strip(),
        "company": (job.get("company_name") or "").strip(),
        "location": job.get("candidate_required_location") or "",
        "remote": True,
        "url": job.get("url"),
        "published_at": _ts(job.get("publication_date")),
        "tags": ", ".join(job.get("tags") or []),
        "description": text,
        "salary_text": job.get("salary") or None,
        "salary_min": smin,
        "salary_max": smax,
        "salary_currency": cur,
        "lang": detect_lang(text),
    }


def normalize_arbeitnow(job: dict) -> dict:
    text = html_to_text(job.get("description"))
    smin, smax, cur = parse_salary(text)
    return {
        "posting_id": f"arbeitnow-{job['slug']}",
        "source": "arbeitnow",
        "title": (job.get("title") or "").strip(),
        "company": (job.get("company_name") or "").strip(),
        "location": job.get("location") or "",
        "remote": bool(job.get("remote")),
        "url": job.get("url"),
        "published_at": _ts(job.get("created_at")),
        "tags": ", ".join(job.get("tags") or []),
        "description": text,
        "salary_text": None,
        "salary_min": smin,
        "salary_max": smax,
        "salary_currency": cur,
        "lang": detect_lang(text),
    }


NORMALIZERS = {"remotive": normalize_remotive, "arbeitnow": normalize_arbeitnow}


def load_raw(raw_dir: Path) -> list[dict]:
    """Read every <source>.json in raw_dir and return normalized postings (not yet deduped)."""
    out: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload.get("source") or path.stem
        fn = NORMALIZERS.get(source)
        if fn is None:
            continue
        for job in payload.get("jobs", []):
            rec = fn(job)
            # Keep data/AI titles only (Remotive's free feed can return any category).
            if rec["title"] and rec["description"] and is_data_title(rec["title"]):
                out.append(rec)
    return out

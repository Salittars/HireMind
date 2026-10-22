"""Fetch data/AI job postings from free public job APIs and cache the raw JSON.

Sources (please respect their terms):
- Remotive  https://remotive.com/api-documentation  -> link back to the Remotive URL and
  name Remotive as the source; fetch at most a few times a day.
- Arbeitnow https://www.arbeitnow.com/blog/job-board-api -> free public API, do not abuse,
  link back to arbeitnow.com.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import RAW_DIR, USER_AGENT

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
REMOTIVE_CATEGORIES = ["data", "artificial-intelligence"]
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

# Arbeitnow is a general job board, so keep only data/AI titles.
DATA_TITLE_RE = re.compile(
    r"\b(data|daten|analyst|analytics|analytiker|machine learning|ml|mlops|ai|ki|genai|llm|"
    r"bi|business intelligence|power bi|scientist|deep learning|nlp|computer vision|"
    r"statistician|quant|dbt)\b",
    re.IGNORECASE,
)
# Titles that match "data" but are not data jobs.
EXCLUDE_TITLE_RE = re.compile(r"\b(data cent(er|re)|rechenzentrum)\b", re.IGNORECASE)


def is_data_title(title: str) -> bool:
    return bool(DATA_TITLE_RE.search(title or "")) and not EXCLUDE_TITLE_RE.search(title or "")


def _get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    for attempt in range(3):
        try:
            r = client.get(url, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_remotive(client: httpx.Client) -> list[dict]:
    jobs: dict[int, dict] = {}
    for cat in REMOTIVE_CATEGORIES:
        payload = _get(client, REMOTIVE_URL, {"category": cat})
        for j in payload.get("jobs", []):
            if is_data_title(j.get("title", "")):
                jobs[j["id"]] = j
        time.sleep(1)
    return list(jobs.values())


def fetch_arbeitnow(client: httpx.Client, max_pages: int = 20, target: int = 600) -> list[dict]:
    kept: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        payload = _get(client, ARBEITNOW_URL, {"page": page})
        data = payload.get("data", [])
        for j in data:
            if is_data_title(j.get("title", "")):
                kept[j["slug"]] = j
        if not data or not payload.get("links", {}).get("next") or len(kept) >= target:
            break
        time.sleep(1.5)  # be polite
    return list(kept.values())


def run(raw_dir: Path = RAW_DIR, max_pages: int = 20, target: int = 600) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for name, fn in [
            ("remotive", lambda: fetch_remotive(client)),
            ("arbeitnow", lambda: fetch_arbeitnow(client, max_pages, target)),
        ]:
            try:
                jobs = fn()
            except Exception as exc:  # one source failing must not kill the run
                print(f"[ingest] {name} failed: {exc}")
                continue
            out = {"source": name, "fetched_at": fetched_at, "jobs": jobs}
            (raw_dir / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False))
            counts[name] = len(jobs)
            print(f"[ingest] {name}: {len(jobs)} postings -> {raw_dir / (name + '.json')}")
    return counts

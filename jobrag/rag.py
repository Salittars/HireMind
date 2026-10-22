"""Retrieve -> (optionally) generate. Works with no API key (extractive answer)."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass, field

import httpx

from .retrieval import Hit
from .skills import extract_skills

SYSTEM_PROMPT = (
    "You answer questions about job postings. Use ONLY the numbered postings given. "
    "Cite every claim with the posting id in square brackets, e.g. [arbeitnow-xyz]. "
    "If the postings do not answer the question, say so. Be concise (max 120 words)."
)


@dataclass
class Answer:
    question: str
    answer: str
    mode: str  # extractive | openai | anthropic
    retriever: str
    sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sources(hits: list[Hit]) -> list[dict]:
    return [
        {"posting_id": h.doc.posting_id, "title": h.doc.title, "company": h.doc.company,
         "url": h.doc.url, "score": round(h.score, 3), "snippet": h.passage}
        for h in hits
    ]


def _context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        d = h.doc
        blocks.append(f"[{d.posting_id}] ({i}) {d.title} at {d.company}, {d.location}\n"
                      f"Skills found: {d.skills}\nText: {d.description[:1500]}")
    return "\n\n".join(blocks)


def extractive_answer(question: str, hits: list[Hit]) -> str:
    """No-LLM answer: skill summary over the retrieved postings + one cited snippet each."""
    if not hits:
        return "No matching postings found."
    counts = Counter(s for h in hits for s in (h.doc.skills.split(", ") if h.doc.skills else []))
    asked = set(extract_skills(question))
    lines = [f"Top {len(hits)} matching postings."]
    if counts:
        common = ", ".join(f"{s} ({n}/{len(hits)})" for s, n in counts.most_common(6))
        lines.append(f"Skills they ask for most: {common}.")
    for s in sorted(asked):
        n = counts.get(s, 0)
        lines.append(f"{s} is mentioned in {n} of {len(hits)} of these postings.")
    for h in hits:
        d = h.doc
        lines.append(f"- [{d.posting_id}] {d.title} – {d.company}: \"{h.passage}\" ({d.url})")
    return "\n".join(lines)


def _openai(question: str, context: str) -> str:
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": f"Postings:\n{context}\n\nQuestion: {question}"}]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _anthropic(question: str, context: str) -> str:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), "max_tokens": 400,
              "system": SYSTEM_PROMPT,
              "messages": [{"role": "user", "content": f"Postings:\n{context}\n\nQuestion: {question}"}]},
        timeout=60,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()


def generator_mode() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "extractive"


def ask(question: str, retriever, k: int = 5) -> Answer:
    hits = retriever.search(question, k=k)
    mode = generator_mode()
    text = extractive_answer(question, hits)
    if mode != "extractive" and hits:
        try:
            fn = _openai if mode == "openai" else _anthropic
            text = fn(question, _context(hits))
        except Exception as exc:  # never fail the request because the LLM call failed
            text = f"(LLM call failed: {type(exc).__name__}; extractive answer below)\n{text}"
            mode = "extractive"
    return Answer(question, text, mode, retriever.name, _sources(hits))

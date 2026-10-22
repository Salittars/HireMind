"""CV -> best matching postings + the skills those postings ask for that the CV lacks."""

from __future__ import annotations

from collections import Counter

from .skills import extract_skills


def match_cv(cv_text: str, retriever, k: int = 10) -> dict:
    cv_skills = set(extract_skills(cv_text))
    hits = retriever.search(cv_text, k=k)
    missing_counter: Counter[str] = Counter()
    matches = []
    for h in hits:
        job_skills = [s for s in h.doc.skills.split(", ") if s] if h.doc.skills else []
        have = [s for s in job_skills if s in cv_skills]
        missing = [s for s in job_skills if s not in cv_skills]
        missing_counter.update(missing)
        matches.append({
            "posting_id": h.doc.posting_id, "title": h.doc.title, "company": h.doc.company,
            "url": h.doc.url, "score": round(h.score, 3),
            "skill_overlap": f"{len(have)}/{len(job_skills)}" if job_skills else "0/0",
            "missing_skills": missing,
        })
    return {
        "retriever": retriever.name,
        "cv_skills": sorted(cv_skills),
        "matches": matches,
        "missing_skills_ranked": [{"skill": s, "in_n_matches": n} for s, n in missing_counter.most_common()],
    }

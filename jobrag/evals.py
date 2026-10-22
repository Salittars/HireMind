"""Retrieval evals on evals/eval_set.jsonl.

Two question types:
- known_item: written by hand after reading one posting, in a job seeker's own words
  (not copied from the title). relevant_ids = that posting (2 ids when two postings fit).
  Metrics: recall@5 (share of relevant ids in the top 5) and MRR (1 / rank of first hit).
- skill: "jobs that need X and Y". A retrieved posting counts as relevant if the regex
  taxonomy finds all expected skills in it. Metrics: precision@5 and MRR.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ROOT
from .retrieval import EmbeddingRetriever, HybridRetriever, TfidfRetriever, load_docs

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
K = 5


def load_eval_set(path: Path = EVAL_SET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recall_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    for i, ok in enumerate(flags, 1):
        if ok:
            return 1.0 / i
    return 0.0


def evaluate(retriever, items: list[dict], doc_ids: set[str], depth: int = 20,
             lang: dict[str, str] | None = None) -> dict:
    lang = lang or {}
    known, skill, per_q = [], [], []
    for it in items:
        hits = retriever.search(it["question"], k=depth)
        ranked = [h.doc.posting_id for h in hits]
        if it["type"] == "known_item":
            rel = set(it["relevant_ids"]) & doc_ids
            if not rel:
                continue  # posting not in this corpus (e.g. running on a different sample)
            r = {"qid": it["qid"], "recall@5": recall_at_k(ranked, rel),
                 "rr": reciprocal_rank([p in rel for p in ranked]),
                 "english": lang.get(sorted(rel)[0], "unknown") == "en"}
            known.append(r)
        else:
            need = set(it["expected_skills"])
            flags = [need <= set(h.doc.skills.split(", ")) for h in hits]
            r = {"qid": it["qid"], "precision@5": sum(flags[:K]) / K, "rr": reciprocal_rank(flags)}
            skill.append(r)
        per_q.append(r)

    def mean(rows, key):
        return round(sum(x[key] for x in rows) / len(rows), 3) if rows else None

    return {
        "retriever": retriever.name,
        "n_known_item": len(known),
        "known_item_recall@5": mean(known, "recall@5"),
        "known_item_mrr": mean(known, "rr"),
        "known_item_recall@5_english_postings": mean([x for x in known if x["english"]], "recall@5"),
        "known_item_recall@5_non_english_postings": mean([x for x in known if not x["english"]], "recall@5"),
        "n_skill": len(skill),
        "skill_precision@5": mean(skill, "precision@5"),
        "skill_mrr": mean(skill, "rr"),
        "per_question": per_q,
    }


def run_evals(kinds: list[str], out: Path | None = None) -> list[dict]:
    docs = load_docs()
    doc_ids = {d.posting_id for d in docs}
    lang = {d.posting_id: d.lang for d in docs}
    items = load_eval_set()
    built: dict[str, object] = {}
    results = []
    for kind in kinds:
        t0 = time.perf_counter()
        if kind == "tfidf":
            r = built.setdefault("tfidf", TfidfRetriever(docs))
        elif kind == "embeddings":
            r = built.setdefault("embeddings", EmbeddingRetriever(docs))
        elif kind == "embeddings-nochunk":
            r = EmbeddingRetriever(docs, chunk=False)
            r.name = "embeddings-nochunk"
        elif kind == "hybrid":
            dense = built.get("embeddings") or EmbeddingRetriever(docs)
            sparse = built.get("tfidf") or TfidfRetriever(docs)
            r = HybridRetriever(docs, dense=dense, sparse=sparse)
        else:
            raise ValueError(kind)
        res = evaluate(r, items, doc_ids, lang=lang)
        res["n_postings"] = len(docs)
        dense = getattr(r, "dense", r)
        res["embed_model"] = getattr(dense, "model_name", None)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        print(f"{res['retriever']:>20} | postings={len(docs)} | known-item recall@5="
              f"{res['known_item_recall@5']} MRR={res['known_item_mrr']} (n={res['n_known_item']}) | "
              f"skill P@5={res['skill_precision@5']} MRR={res['skill_mrr']} (n={res['n_skill']}) | "
              f"recall@5 en={res['known_item_recall@5_english_postings']} "
              f"non-en={res['known_item_recall@5_non_english_postings']}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
    return results

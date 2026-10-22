"""Vector search over postings.

Two retrievers with the same interface:
- EmbeddingRetriever: sentence-transformers (multilingual-e5-small, CPU) + cosine similarity in numpy.
  Long descriptions are split into overlapping word chunks; a posting scores as its best chunk.
- TfidfRetriever: scikit-learn TF-IDF (word 1-2 grams) + cosine. Needs no download; it is the
  baseline in the evals and the fallback when the model cannot be loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DB_PATH, EMBED_MODEL, EMBEDDER, INDEX_DIR


@dataclass
class Doc:
    posting_id: str
    title: str
    company: str
    location: str
    url: str
    role_family: str
    skills: str
    description: str
    lang: str = "unknown"

    @property
    def text(self) -> str:
        return f"{self.title}. {self.company}. {self.location}. {self.description}"


@dataclass
class Hit:
    doc: Doc
    score: float
    passage: str


def load_docs(db_path: Path = DB_PATH) -> list[Doc]:
    from .warehouse import connect

    con = connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            select m.posting_id, m.title, m.company, coalesce(m.location, ''), coalesce(m.url, ''),
                   m.role_family, coalesce(m.skills, ''), s.description, coalesce(m.lang, 'unknown')
            from mart_postings m join stg_postings s using (posting_id)
            order by m.posting_id
            """
        ).fetchall()
    finally:
        con.close()
    return [Doc(*r) for r in rows]


def chunk_words(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping chunks of `size` words (MiniLM reads ~256 tokens max)."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, docs: list[Doc]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.9, strip_accents="unicode"
        )
        self.matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def _scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._scores(query)
        top = np.argsort(-scores)[:k]
        return [Hit(self.docs[i], float(scores[i]), best_passage(self.docs[i], query)) for i in top]


class EmbeddingRetriever:
    name = "embeddings"

    def __init__(self, docs: list[Doc], model_name: str = EMBED_MODEL, index_dir: Path = INDEX_DIR,
                 chunk: bool = True, max_seconds: float | None = None):
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.chunk = chunk
        self.max_seconds = max_seconds
        self.chunks: list[str] = []
        owner: list[int] = []
        for i, d in enumerate(docs):
            parts = chunk_words(d.text) if chunk else [d.text]
            self.chunks += parts
            owner += [i] * len(parts)
        self.owner = np.array(owner)
        passage_prefix = self._prefixes[1]
        self.matrix = self._embed_cached([passage_prefix + c for c in self.chunks], index_dir)

    @property
    def _prefixes(self) -> tuple[str, str]:
        """E5 models are trained with 'query: ' / 'passage: ' prefixes; others use none."""
        return ("query: ", "passage: ") if "e5" in self.model_name.lower() else ("", "")

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    def _embed_cached(self, texts: list[str], index_dir: Path) -> np.ndarray:
        """Embeddings are cached per chunk hash, so re-running only embeds new text."""
        index_dir.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^a-zA-Z0-9]+", "_", self.model_name)
        vec_path, key_path = index_dir / f"{tag}.npy", index_dir / f"{tag}.keys.json"
        cache: dict[str, np.ndarray] = {}
        if vec_path.exists() and key_path.exists():
            keys = json.loads(key_path.read_text())
            vecs = np.load(vec_path)
            cache = dict(zip(keys, vecs, strict=False))
        hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [i for i, h in enumerate(hashes) if h not in cache]
        deadline = time.monotonic() + self.max_seconds if self.max_seconds else None

        def save() -> None:
            keys = list(cache)
            np.save(vec_path, np.stack([cache[k] for k in keys]))
            key_path.write_text(json.dumps(keys))

        # Embed in blocks and save after each one, so a long first build can be resumed.
        for start in range(0, len(missing), 256):
            block = missing[start : start + 256]
            for i, v in zip(block, self._embed([texts[i] for i in block]), strict=True):
                cache[hashes[i]] = v
            save()
            if deadline and time.monotonic() > deadline and start + 256 < len(missing):
                done = sum(h in cache for h in hashes)
                raise TimeoutError(f"embedded {done}/{len(texts)} chunks; run again to resume")
        return np.stack([cache[h] for h in hashes])

    def encode_query(self, text: str) -> np.ndarray:
        """Long inputs (e.g. a CV) are chunked and averaged."""
        parts = [self._prefixes[0] + p for p in chunk_words(text)]
        return _normalize(self._embed(parts).mean(axis=0, keepdims=True))[0]

    def _scores(self, query: str) -> np.ndarray:
        q = self.encode_query(query)
        chunk_scores = self.matrix @ q
        scores = np.full(len(self.docs), -1.0, dtype=np.float32)
        np.maximum.at(scores, self.owner, chunk_scores)  # posting score = best chunk
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._scores(query)
        top = np.argsort(-scores)[:k]
        return [Hit(self.docs[i], float(scores[i]), best_passage(self.docs[i], query)) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion of TF-IDF (exact keywords) and embeddings (meaning).

    score(doc) = sum over retrievers of 1 / (rrf_k + rank). No score scaling needed.
    """

    name = "hybrid"

    def __init__(self, docs: list[Doc], rrf_k: int = 60, dense: EmbeddingRetriever | None = None,
                 sparse: TfidfRetriever | None = None):
        self.docs = docs
        self.rrf_k = rrf_k
        self.dense = dense or EmbeddingRetriever(docs)
        self.sparse = sparse or TfidfRetriever(docs)

    def _scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.docs))
        for scores in (self.dense._scores(query), self.sparse._scores(query)):
            ranks = np.empty(len(scores), dtype=int)
            ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
            fused += 1.0 / (self.rrf_k + ranks)
        return fused

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._scores(query)
        top = np.argsort(-scores)[:k]
        return [Hit(self.docs[i], float(scores[i]), best_passage(self.docs[i], query)) for i in top]


_SENT_SPLIT = re.compile(r"(?<=[.!?•;:])\s+")
_TOKEN = re.compile(r"[a-z0-9+#/]+")
_STOP = set("a an the and or of to in for with on at by is are be as we you our your what which "
            "who how do does job jobs role roles posting postings ask asks need needs want".split())


def _passages(text: str, max_words: int = 45) -> list[str]:
    """Sentences; long ones (bullet lists flattened by HTML cleaning) cut into word windows."""
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        words = sent.split()
        if len(words) <= max_words:
            out.append(sent)
        else:
            step = max_words - 10
            out += [" ".join(words[i : i + max_words]) for i in range(0, len(words) - 10, step)]
    return out


def best_passage(doc: Doc, query: str, max_chars: int = 280) -> str:
    """Extractive snippet: the description passage sharing the most query words."""
    q = {t for t in _TOKEN.findall(query.lower()) if t not in _STOP}
    best, best_score = "", -1.0
    for sent in _passages(doc.description):
        if len(sent) < 25:
            continue
        toks = set(_TOKEN.findall(sent.lower()))
        score = len(q & toks) + 0.001 * min(len(sent), 200)
        if score > best_score:
            best, best_score = sent, score
    best = best or doc.description
    if len(best) <= max_chars:
        return best
    # Too long: centre the snippet on the first query word it contains.
    low = best.lower()
    pos = min((m.start() for t in q for m in [re.search(rf"\b{re.escape(t)}\b", low)] if m), default=0)
    start = max(0, min(pos - max_chars // 3, len(best) - max_chars))
    snippet = best[start : start + max_chars]
    if start > 0:
        snippet = "…" + snippet.split(" ", 1)[-1]
    if start + max_chars < len(best):
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    return snippet


def get_retriever(docs: list[Doc] | None = None, kind: str = EMBEDDER):
    """kind: auto | hybrid | sentence-transformers | tfidf.

    auto = hybrid (best in the evals) and falls back to TF-IDF if the model cannot load.
    """
    docs = docs if docs is not None else load_docs()
    if kind in ("sentence-transformers", "embeddings"):
        return EmbeddingRetriever(docs)
    if kind in ("auto", "hybrid"):
        try:
            return HybridRetriever(docs)
        except Exception as exc:
            if kind != "auto":
                raise
            print(f"[retrieval] embeddings unavailable ({type(exc).__name__}: {exc}); using TF-IDF")
    return TfidfRetriever(docs)

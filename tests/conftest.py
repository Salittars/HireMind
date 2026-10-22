"""Shared fixtures. Everything runs offline on data/sample + tests/fixtures, with TF-IDF."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

os.environ.setdefault("JOBRAG_EMBEDDER", "tfidf")
for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(key, None)  # tests must never call a paid API


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> Path:
    from jobrag.warehouse import build

    db = tmp_path_factory.mktemp("wh") / "sample.duckdb"
    build(raw_dir=SAMPLE, db_path=db)
    return db


@pytest.fixture(scope="session")
def sample_docs(sample_db):
    from jobrag.retrieval import load_docs

    return load_docs(sample_db)


@pytest.fixture(scope="session")
def tfidf(sample_docs):
    from jobrag.retrieval import TfidfRetriever

    return TfidfRetriever(sample_docs)

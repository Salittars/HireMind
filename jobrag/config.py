"""Paths and settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default))).resolve()


DATA_DIR = _path("JOBRAG_DATA_DIR", ROOT / "data")
# Where raw API JSON lives. Point this at data/sample to run fully offline.
RAW_DIR = _path("JOBRAG_RAW_DIR", DATA_DIR / "raw")
SAMPLE_DIR = ROOT / "data" / "sample"
DB_PATH = _path("JOBRAG_DB", DATA_DIR / "warehouse.duckdb")
INDEX_DIR = _path("JOBRAG_INDEX_DIR", DATA_DIR / "index")
MODELS_DIR = ROOT / "models"
SEEDS_DIR = ROOT / "seeds"
TAXONOMY_CSV = SEEDS_DIR / "skill_taxonomy.csv"

# "auto" = hybrid (TF-IDF + embeddings) if sentence-transformers loads, else TF-IDF only.
EMBEDDER = os.environ.get("JOBRAG_EMBEDDER", "auto")
EMBED_MODEL = os.environ.get("JOBRAG_EMBED_MODEL", "intfloat/multilingual-e5-small")

USER_AGENT = "hiremind/0.1 (portfolio project; https://github.com/Salittars/HireMind)"

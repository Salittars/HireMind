"""Transparent skill taxonomy and role families, both stored as CSV seeds.

The same CSVs are used by Python (CV matching, tests) and by DuckDB SQL (the marts), so
there is exactly one definition of "this posting asks for skill X".
Patterns are written in the subset of regex shared by Python `re` and DuckDB/RE2,
and are always applied to lower-cased text.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import SEEDS_DIR, TAXONOMY_CSV

ROLE_FAMILIES_CSV = SEEDS_DIR / "role_families.csv"
OTHER_ROLE = "Other data role"


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    pattern: str


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=4)
def load_taxonomy(path: Path = TAXONOMY_CSV) -> tuple[Skill, ...]:
    return tuple(Skill(r["skill"], r["category"], r["pattern"]) for r in _read_csv(path))


@lru_cache(maxsize=4)
def _compiled(path: Path = TAXONOMY_CSV) -> tuple[tuple[str, re.Pattern], ...]:
    return tuple((s.name, re.compile(s.pattern)) for s in load_taxonomy(path))


def extract_skills(text: str) -> list[str]:
    """Return skill names found in text (case-insensitive), in taxonomy order."""
    low = (text or "").lower()
    return [name for name, rx in _compiled() if rx.search(low)]


@lru_cache(maxsize=1)
def _role_rules() -> tuple[tuple[str, re.Pattern], ...]:
    rows = sorted(_read_csv(ROLE_FAMILIES_CSV), key=lambda r: int(r["priority"]))
    return tuple((r["role_family"], re.compile(r["pattern"])) for r in rows)


def role_family(title: str) -> str:
    """First matching rule (by priority) on the lower-cased title wins."""
    low = (title or "").lower()
    for fam, rx in _role_rules():
        if rx.search(low):
            return fam
    return OTHER_ROLE

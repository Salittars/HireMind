"""Write the small committed sample (data/sample/) used by tests, CI and the offline demo.

The sample keeps Arbeitnow postings only (their API terms ask for a link back, which every
record keeps in `url`). Remotive's terms ask that their jobs are not redistributed to third
parties, so Remotive records stay in the local, git-ignored cache. The sample always contains
every posting referenced by evals/eval_set.jsonl, plus a spread of other postings.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import RAW_DIR, ROOT, SAMPLE_DIR

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
MAX_BYTES = 950_000


def _eval_ids() -> set[str]:
    ids: set[str] = set()
    if EVAL_SET.exists():
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.update(json.loads(line).get("relevant_ids", []))
    return ids


def write_sample(raw_dir: Path = RAW_DIR, out_dir: Path = SAMPLE_DIR, n_extra: int = 120) -> Path:
    payload = json.loads((raw_dir / "arbeitnow.json").read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    wanted = {i.removeprefix("arbeitnow-") for i in _eval_ids()}
    chosen = [j for j in jobs if j["slug"] in wanted]
    missing = wanted - {j["slug"] for j in chosen}
    if missing:
        raise SystemExit(f"eval postings missing from raw data: {sorted(missing)[:5]}")
    others = sorted((j for j in jobs if j["slug"] not in wanted), key=lambda j: j["slug"])
    step = max(1, len(others) // n_extra)
    for j in others[::step]:
        trial = chosen + [j]
        if len(json.dumps({"jobs": trial}, ensure_ascii=False).encode()) > MAX_BYTES:
            break
        chosen = trial
    out = {
        "source": "arbeitnow",
        "fetched_at": payload["fetched_at"],
        "attribution": "Job data from the Arbeitnow free job board API (https://www.arbeitnow.com). "
                       "Each posting links back to its original page.",
        "jobs": chosen,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "arbeitnow.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[sample] {len(chosen)} postings, {path.stat().st_size / 1e6:.2f} MB -> {path}")
    return path

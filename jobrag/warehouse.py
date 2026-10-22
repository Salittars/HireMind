"""Load clean postings + seeds into DuckDB and run the SQL models (a tiny dbt-style runner).

Models live in models/<layer>/<name>.sql. Each file is one SELECT; the runner turns it
into `CREATE OR REPLACE TABLE <name> AS ...`. Layers run in order:
staging -> intermediate -> marts, files alphabetically inside a layer.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from .clean import load_raw
from .config import DB_PATH, MODELS_DIR, RAW_DIR, SEEDS_DIR

LAYERS = ["staging", "intermediate", "marts"]

RAW_COLUMNS = [
    "posting_id", "source", "title", "company", "location", "remote", "url", "published_at",
    "tags", "description", "salary_text", "salary_min", "salary_max", "salary_currency", "lang",
]


def connect(db_path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def load_seeds(con: duckdb.DuckDBPyConnection, seeds_dir: Path = SEEDS_DIR) -> list[str]:
    names = []
    for path in sorted(seeds_dir.glob("*.csv")):
        name = f"seed_{path.stem}"
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_csv(?, header=true, all_varchar=false, quote='\"')",
            [str(path)],
        )
        names.append(name)
    return names


def load_raw_postings(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    df = pd.DataFrame(records, columns=RAW_COLUMNS)
    df["salary_min"] = pd.to_numeric(df["salary_min"])
    df["salary_max"] = pd.to_numeric(df["salary_max"])
    con.register("raw_df", df)
    con.execute("CREATE OR REPLACE TABLE raw_postings AS SELECT * FROM raw_df")
    con.unregister("raw_df")
    return len(df)


def model_files(models_dir: Path = MODELS_DIR) -> list[Path]:
    files: list[Path] = []
    for layer in LAYERS:
        files += sorted((models_dir / layer).glob("*.sql"))
    return files


def run_models(con: duckdb.DuckDBPyConnection, models_dir: Path = MODELS_DIR) -> dict[str, int]:
    rows: dict[str, int] = {}
    for path in model_files(models_dir):
        sql = path.read_text(encoding="utf-8").strip().rstrip(";")
        t0 = time.perf_counter()
        con.execute(f"CREATE OR REPLACE TABLE {path.stem} AS {sql}")
        n = con.execute(f"SELECT count(*) FROM {path.stem}").fetchone()[0]
        rows[path.stem] = n
        print(f"[models] {path.parent.name}/{path.stem}: {n} rows ({time.perf_counter() - t0:.2f}s)")
    return rows


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    records = load_raw(raw_dir)
    if not records:
        raise SystemExit(f"No raw postings in {raw_dir}. Run `python -m jobrag ingest` "
                         f"or set JOBRAG_RAW_DIR=data/sample.")
    con = connect(db_path)
    try:
        load_seeds(con)
        n_raw = load_raw_postings(con, records)
        print(f"[load] raw_postings: {n_raw} rows from {raw_dir}")
        out = {"raw_postings": n_raw, **run_models(con)}
    finally:
        con.close()
    return out

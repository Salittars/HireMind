"""Command line: python -m jobrag <command>

  ingest            fetch live postings from the public APIs into data/raw/
  build             clean + load into DuckDB + run SQL models
  index             build (or resume) the embedding index
  ask "question"    retrieve postings and answer with citations
  match cv.txt      match a CV (text file, or - for stdin) to postings
  report            print the skill marts
  chart             write docs/top_skills.png from the marts
  eval              retrieval evals (recall@5, MRR): embeddings vs TF-IDF
  sample            write the small committed sample to data/sample/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def _retriever(kind: str | None):
    from .retrieval import get_retriever

    return get_retriever(kind=kind or config.EMBEDDER)


def cmd_ingest(a) -> None:
    from .ingest import run

    run(max_pages=a.max_pages, target=a.target)


def cmd_build(a) -> None:
    from .warehouse import build

    build()


def cmd_index(a) -> None:
    from .retrieval import EmbeddingRetriever, load_docs

    try:
        r = EmbeddingRetriever(load_docs(), max_seconds=a.max_seconds)
        print(f"[index] {len(r.docs)} postings, {r.matrix.shape[0]} chunks embedded with {r.model_name}")
    except TimeoutError as exc:
        print(f"[index] {exc}")


def cmd_ask(a) -> None:
    from .rag import ask

    ans = ask(a.question, _retriever(a.retriever), k=a.k)
    if a.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Q: {ans.question}\n[retriever={ans.retriever}, generator={ans.mode}]\n\n{ans.answer}")


def cmd_match(a) -> None:
    from .match import match_cv

    text = sys.stdin.read() if a.cv == "-" else Path(a.cv).read_text(encoding="utf-8")
    print(json.dumps(match_cv(text, _retriever(a.retriever), k=a.k), indent=2, ensure_ascii=False))


def cmd_report(a) -> None:
    from .warehouse import connect

    con = connect(read_only=True)
    for title, sql in [
        ("Postings", "select count(*) as postings, count(distinct company) as companies from mart_postings"),
        ("Top skills", "select skill, n_postings, share_pct from mart_skill_frequency limit 15"),
        ("Role families", "select role_family, count(*) n from mart_postings group by 1 order by 2 desc"),
        ("Top skill pairs", "select skill_a, skill_b, n_together, lift from mart_skill_cooccurrence limit 10"),
        ("Salary (where stated)", "select * from mart_salary limit 10"),
    ]:
        print(f"\n== {title} ==")
        print(con.sql(sql))
    con.close()


def cmd_chart(a) -> None:
    from .charts import top_skills_chart

    print(top_skills_chart(Path(a.out)))


def cmd_eval(a) -> None:
    from .evals import run_evals

    run_evals(kinds=a.retrievers.split(","), out=Path(a.out) if a.out else None)


def cmd_sample(a) -> None:
    from .sample import write_sample

    write_sample()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jobrag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--max-pages", type=int, default=20)
    s.add_argument("--target", type=int, default=700, help="stop once this many data/AI postings are kept")
    s.set_defaults(fn=cmd_ingest)

    sub.add_parser("build").set_defaults(fn=cmd_build)

    s = sub.add_parser("index")
    s.add_argument("--max-seconds", type=float, default=None, help="stop early; rerun to resume")
    s.set_defaults(fn=cmd_index)

    for name, fn in [("ask", cmd_ask), ("match", cmd_match)]:
        s = sub.add_parser(name)
        s.add_argument("question" if name == "ask" else "cv")
        s.add_argument("-k", type=int, default=5 if name == "ask" else 10)
        s.add_argument("--retriever", choices=["auto", "sentence-transformers", "tfidf"], default=None)
        if name == "ask":
            s.add_argument("--json", action="store_true")
        s.set_defaults(fn=fn)

    sub.add_parser("report").set_defaults(fn=cmd_report)

    s = sub.add_parser("chart")
    s.add_argument("--out", default=str(config.ROOT / "docs" / "top_skills.png"))
    s.set_defaults(fn=cmd_chart)

    s = sub.add_parser("eval")
    s.add_argument("--retrievers", default="tfidf,embeddings,hybrid")
    s.add_argument("--out", default=None, help="write results JSON here")
    s.set_defaults(fn=cmd_eval)

    sub.add_parser("sample").set_defaults(fn=cmd_sample)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

from fastapi.testclient import TestClient

from jobrag import api, rag
from jobrag.evals import evaluate, load_eval_set, recall_at_k, reciprocal_rank
from jobrag.match import match_cv
from jobrag.retrieval import best_passage, chunk_words


def test_chunk_words_overlap():
    words = " ".join(str(i) for i in range(300))
    chunks = chunk_words(words, size=120, overlap=30)
    assert chunks[0].split()[0] == "0" and chunks[1].split()[0] == "90"
    assert chunks[-1].split()[-1] == "299"
    assert chunk_words("short text") == ["short text"]


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, k=5) == 0.5
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([False]) == 0.0


def test_tfidf_finds_known_posting(tfidf):
    hits = tfidf.search("own the dbt transformation layer at a security awareness training company", k=5)
    assert "arbeitnow-remote-senior-analytics-engineer-83455" in [h.doc.posting_id for h in hits]


def test_eval_set_runs_on_sample(tfidf, sample_docs):
    items = load_eval_set()
    assert len(items) >= 30
    res = evaluate(tfidf, items, {d.posting_id for d in sample_docs})
    assert res["n_known_item"] >= 30  # every hand-labelled posting is in the committed sample
    assert res["known_item_recall@5"] >= 0.6
    assert res["skill_precision@5"] >= 0.5


def test_best_passage_prefers_query_terms(sample_docs):
    doc = next(d for d in sample_docs if "dbt" in d.description.lower())
    assert "dbt" in best_passage(doc, "dbt").lower()


def test_extractive_answer_cites_sources_without_key(tfidf):
    ans = rag.ask("Which jobs ask for dbt and Snowflake?", tfidf, k=5)
    assert ans.mode == "extractive"
    assert len(ans.sources) == 5
    for s in ans.sources:
        assert f"[{s['posting_id']}]" in ans.answer
        assert s["url"].startswith("http")


def test_match_cv_reports_missing_skills(tfidf):
    cv = "Junior BI analyst. Python, SQL, Power BI, Excel dashboards and data modelling."
    out = match_cv(cv, tfidf, k=5)
    assert {"Python", "SQL", "Power BI", "Excel"} <= set(out["cv_skills"])
    assert len(out["matches"]) == 5
    for m in out["matches"]:
        assert not set(m["missing_skills"]) & set(out["cv_skills"])


def test_api_endpoints(tfidf, monkeypatch):
    monkeypatch.setattr(api, "retriever", lambda: tfidf)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "Power BI dashboard roles", "k": 3})
    assert r.status_code == 200 and len(r.json()["sources"]) == 3
    r = client.post("/match", json={"cv_text": "Python SQL Airflow dbt data engineer with Docker", "k": 3})
    assert r.status_code == 200 and "missing_skills_ranked" in r.json()
    assert client.post("/ask", json={"question": ""}).status_code == 422

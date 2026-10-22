import re

import duckdb

from jobrag.skills import extract_skills, load_taxonomy, role_family


def test_taxonomy_loads_and_patterns_compile():
    tax = load_taxonomy()
    assert len(tax) >= 40
    names = [s.name for s in tax]
    assert len(names) == len(set(names))
    for s in tax:
        re.compile(s.pattern)


def test_extract_core_skills():
    text = "We use Python, SQL and dbt on Snowflake; dashboards in Power BI. Deploy with Docker + GitHub Actions."
    found = set(extract_skills(text))
    assert {"Python", "SQL", "dbt", "Snowflake", "Power BI", "Docker", "CI/CD", "Git"} <= found


def test_extract_ai_skills():
    found = set(extract_skills("Build retrieval-augmented generation (RAG) apps with LLMs, LangChain and pgvector."))
    assert {"RAG", "LLM", "LangChain/LlamaIndex", "Vector databases"} <= found


def test_no_false_positives_on_lookalike_words():
    # PostgreSQL is not "SQL", "excellent" is not Excel, "Vertrag" is not RAG, "R&D" is not R.
    found = set(extract_skills("Excellent PostgreSQL skills, R&D background, unbefristeter Vertrag."))
    assert "SQL" not in found and "Excel" not in found and "RAG" not in found and "R" not in found
    assert "PostgreSQL" in found


def test_python_and_duckdb_regex_agree(sample_db):
    """The SQL marts and the Python matcher must find the same skills for every posting."""
    con = duckdb.connect(str(sample_db), read_only=True)
    sql_pairs = set(con.execute("select posting_id, skill from int_posting_skills").fetchall())
    rows = con.execute("select posting_id, search_text from stg_postings").fetchall()
    con.close()
    py_pairs = {(pid, s) for pid, text in rows for s in extract_skills(text)}
    assert sql_pairs == py_pairs


def test_role_family():
    assert role_family("Senior Analytics Engineer") == "Analytics engineer"
    assert role_family("Data Engineer (m/w/d)") == "Data engineer"
    assert role_family("Senior Data Scientist") == "Data scientist"
    assert role_family("Machine Learning Engineer") == "ML / AI role"
    assert role_family("Power BI Developer") == "BI / data analyst"
    assert role_family("Senior Business Analyst") == "Business analyst"
    assert role_family("Data Governance Lead") == "Other data role"

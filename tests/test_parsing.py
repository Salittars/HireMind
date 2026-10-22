import json

from jobrag.clean import detect_lang, html_to_text, load_raw, normalize_arbeitnow, parse_salary
from jobrag.ingest import is_data_title

from .conftest import FIXTURES, SAMPLE


def test_html_to_text_strips_tags_and_entities():
    assert html_to_text("<p>Python &amp; SQL</p><ul><li>dbt</li></ul>") == "Python & SQL dbt"
    assert html_to_text(None) == ""


def test_html_to_text_handles_double_escaped_html():
    # Some feeds escape their HTML twice; one parse pass would leave "<p>" in the text.
    assert html_to_text("&lt;p&gt;Build &lt;strong&gt;RAG&lt;/strong&gt;&lt;/p&gt;") == "Build RAG"


def test_sample_descriptions_have_no_leftover_tags():
    records = load_raw(SAMPLE)
    assert len(records) >= 100
    assert not [r["posting_id"] for r in records if "</p>" in r["description"] or "<br" in r["description"]]


def test_parse_salary_ranges_and_currencies():
    assert parse_salary("$120k - $150k") == (120000.0, 150000.0, "USD")
    assert parse_salary("Gehalt: 65.000 - 80.000 € brutto") == (65000.0, 80000.0, "EUR")
    assert parse_salary("£55,000 per year") == (55000.0, 55000.0, "GBP")


def test_parse_salary_ignores_numbers_without_currency_or_implausible():
    assert parse_salary("We have 12,000 customers and 300 employees") == (None, None, None)
    assert parse_salary("Raised €25,000,000 in funding") == (None, None, None)
    assert parse_salary("") == (None, None, None)


def test_detect_lang():
    assert detect_lang("We are looking for you to join our team and build the future") == "en"
    assert detect_lang("Wir suchen für unser Team eine Person mit Erfahrung und die Lust hat") == "de"


def test_data_title_filter():
    assert is_data_title("Senior Data Engineer (m/w/d)")
    assert is_data_title("Machine Learning Engineer")
    assert is_data_title("BI Analyst")
    assert not is_data_title("Customer Support Agent")
    assert not is_data_title("Directeur de Projet DATA CENTER")


def test_remotive_fixture_is_normalized_and_filtered():
    records = load_raw(FIXTURES)
    ids = [r["posting_id"] for r in records]
    assert "remotive-1003" not in ids  # not a data title
    r = next(r for r in records if r["posting_id"] == "remotive-1001")
    assert r["source"] == "remotive" and r["remote"] is True
    assert r["salary_min"] == 120000 and r["salary_currency"] == "USD"
    assert "<" not in r["description"] and "dbt" in r["description"]


def test_arbeitnow_normalizer_keeps_link_back():
    job = json.loads((SAMPLE / "arbeitnow.json").read_text())["jobs"][0]
    rec = normalize_arbeitnow(job)
    assert rec["posting_id"] == f"arbeitnow-{job['slug']}"
    assert rec["url"].startswith("https://www.arbeitnow.")
    assert not rec["description"].endswith("on Arbeitnow")  # footer boilerplate removed

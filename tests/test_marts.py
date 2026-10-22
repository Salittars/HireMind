import duckdb
import pytest

from jobrag.warehouse import build, model_files

from .conftest import FIXTURES


@pytest.fixture(scope="module")
def con(sample_db):
    c = duckdb.connect(str(sample_db), read_only=True)
    yield c
    c.close()


def test_models_run_in_layer_order():
    layers = [p.parent.name for p in model_files()]
    assert layers == sorted(layers, key=["staging", "intermediate", "marts"].index)


def test_staging_has_unique_ids_and_no_duplicate_title_company(con):
    n, n_ids, n_tc = con.execute(
        "select count(*), count(distinct posting_id), count(distinct "
        "regexp_replace(lower(title), '[^a-z0-9]', '', 'g') || '|' || "
        "regexp_replace(lower(company), '[^a-z0-9]', '', 'g')) from stg_postings"
    ).fetchone()
    assert n == n_ids == n_tc


def test_dedupe_drops_cross_posts(tmp_path):
    out = build(raw_dir=FIXTURES, db_path=tmp_path / "fx.duckdb")
    assert out["raw_postings"] == 2  # the support job is filtered at load time
    assert out["stg_postings"] == 1  # same title + company counts once


def test_skill_frequency_shares_are_consistent(con):
    total = con.execute("select count(*) from stg_postings").fetchone()[0]
    rows = con.execute("select skill, n_postings, share_pct from mart_skill_frequency").fetchall()
    assert rows, "mart is empty"
    for skill, n, share in rows:
        assert 0 < n <= total
        assert share == pytest.approx(round(100 * n / total, 1)), skill
    counts = [r[1] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_cooccurrence_pairs_are_ordered_and_bounded(con):
    bad = con.execute(
        """
        select count(*) from mart_skill_cooccurrence c
        join mart_skill_frequency a on a.skill = c.skill_a
        join mart_skill_frequency b on b.skill = c.skill_b
        where c.skill_a >= c.skill_b or c.n_together > least(a.n_postings, b.n_postings)
              or c.jaccard < 0 or c.jaccard > 1
        """
    ).fetchone()[0]
    assert bad == 0


def test_every_posting_has_one_role_family(con):
    n, n_roles = con.execute(
        "select (select count(*) from stg_postings), (select count(distinct posting_id) from int_posting_roles)"
    ).fetchone()
    assert n == n_roles
    fam_total = con.execute(
        "select sum(n_family) from (select distinct role_family, n_family from mart_skills_by_role_family)"
    ).fetchone()[0]
    assert fam_total <= n


def test_salary_mart_only_plausible_values(con):
    rows = con.execute("select lowest_min, highest_max from mart_salary").fetchall()
    for lo, hi in rows:
        assert 10_000 <= lo <= hi <= 400_000

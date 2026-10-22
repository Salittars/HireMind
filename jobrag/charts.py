"""Chart of the most-demanded skills, read from the mart (so it always matches the data)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .warehouse import connect  # noqa: E402

CATEGORY_COLORS = {
    "language": "#1f4e79", "ai": "#c0392b", "ml": "#e67e22", "devops": "#7f8c8d",
    "cloud": "#2e86c1", "data engineering": "#27ae60", "bi": "#8e44ad", "database": "#16a085",
}


def top_skills_chart(out: Path, n: int = 15) -> Path:
    con = connect(read_only=True)
    try:
        total = con.execute("select count(*) from mart_postings").fetchone()[0]
        rows = con.execute(
            "select skill, category, n_postings, share_pct from mart_skill_frequency "
            "order by n_postings desc, skill limit ?", [n]
        ).fetchall()
        first, last = con.execute(
            "select strftime(min(published_at), '%Y-%m-%d'), strftime(max(published_at), '%Y-%m-%d') "
            "from mart_postings"
        ).fetchone()
    finally:
        con.close()
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    bars = ax.barh([r[0] for r in rows], [r[3] for r in rows],
                   color=[CATEGORY_COLORS.get(r[1], "#555") for r in rows])
    for bar, r in zip(bars, rows, strict=True):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{r[3]:.1f}%",
                va="center", fontsize=9)
    ax.set_xlabel("% of postings mentioning the skill")
    ax.set_title(f"Top {n} skills in {total} data/AI job postings ({first} to {last})", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(r[3] for r in rows) * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), loc="lower right", fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Source: Arbeitnow + Remotive public job APIs. Skill = regex match "
             "(seeds/skill_taxonomy.csv).", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

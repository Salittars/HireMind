-- One row per (posting, skill) found by the regex taxonomy in seeds/skill_taxonomy.csv.
select
    p.posting_id,
    t.skill,
    t.category
from stg_postings as p
inner join seed_skill_taxonomy as t
    on regexp_matches(p.search_text, t.pattern)

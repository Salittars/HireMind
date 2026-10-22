-- Role family from the title, using seeds/role_families.csv (lowest priority number wins).
select
    p.posting_id,
    coalesce(arg_min(r.role_family, r.priority), 'Other data role') as role_family
from stg_postings as p
left join seed_role_families as r
    on regexp_matches(lower(p.title), r.pattern)
group by p.posting_id

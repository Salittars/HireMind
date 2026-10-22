-- Skill share inside each role family (e.g. % of data engineer ads asking for Spark).
with family_size as (
    select role_family, count(*) as n_family
    from int_posting_roles
    group by role_family
)

select
    r.role_family,
    f.n_family,
    s.skill,
    count(*) as n_postings,
    round(100.0 * count(*) / f.n_family, 1) as share_in_family_pct
from int_posting_skills as s
inner join int_posting_roles as r using (posting_id)
inner join family_size as f using (role_family)
group by r.role_family, f.n_family, s.skill
order by r.role_family, n_postings desc, s.skill

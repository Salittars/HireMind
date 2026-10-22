-- Posting dimension: staging + role family + number of skills found.
select
    p.posting_id,
    p.source,
    p.title,
    p.company,
    p.location,
    p.remote,
    p.url,
    p.published_at,
    p.lang,
    r.role_family,
    count(s.skill) as n_skills,
    string_agg(s.skill, ', ' order by s.skill) as skills,
    p.salary_min,
    p.salary_max,
    p.salary_currency
from stg_postings as p
inner join int_posting_roles as r using (posting_id)
left join int_posting_skills as s using (posting_id)
group by all

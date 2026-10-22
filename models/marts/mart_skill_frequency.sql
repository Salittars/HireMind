-- How many postings mention each skill, and the share of all postings.
with total as (select count(*) as n from stg_postings)

select
    s.skill,
    s.category,
    count(distinct s.posting_id) as n_postings,
    round(100.0 * count(distinct s.posting_id) / total.n, 1) as share_pct,
    rank() over (order by count(distinct s.posting_id) desc) as skill_rank
from int_posting_skills as s
cross join total
group by s.skill, s.category, total.n
order by n_postings desc, s.skill

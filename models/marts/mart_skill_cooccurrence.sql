-- Pairs of skills asked for in the same posting. lift > 1 = appear together more than by chance.
with total as (select count(*) as n from stg_postings),

freq as (
    select skill, count(*) as n_skill
    from int_posting_skills
    group by skill
),

pairs as (
    select a.skill as skill_a, b.skill as skill_b, count(*) as n_together
    from int_posting_skills as a
    inner join int_posting_skills as b
        on a.posting_id = b.posting_id and a.skill < b.skill
    group by a.skill, b.skill
)

select
    p.skill_a,
    p.skill_b,
    p.n_together,
    round(p.n_together * 1.0 / (fa.n_skill + fb.n_skill - p.n_together), 3) as jaccard,
    round(p.n_together * total.n * 1.0 / (fa.n_skill * fb.n_skill), 2) as lift
from pairs as p
inner join freq as fa on p.skill_a = fa.skill
inner join freq as fb on p.skill_b = fb.skill
cross join total
where p.n_together >= 3
order by p.n_together desc, p.skill_a, p.skill_b

-- Salary where the posting states one (yearly, with an explicit currency).
select
    r.role_family,
    p.salary_currency as currency,
    count(*) as n_with_salary,
    median(p.salary_min) as median_salary_min,
    median(p.salary_max) as median_salary_max,
    min(p.salary_min) as lowest_min,
    max(p.salary_max) as highest_max
from stg_postings as p
inner join int_posting_roles as r using (posting_id)
where p.salary_min is not null
group by r.role_family, p.salary_currency
order by n_with_salary desc
